"""手製クロップの枠を逆算して registry.json に書き戻す
(scaling-verification 第0段 / figure-fetch-distribution Phase 0・§3.5)。

背景: verified 139件のうち 123件は `data/verified_pairs/crops/` 配下の
手製クロップを指しており、**どの元画像のどの矩形か・回転は何度かが
どこにも記録されていない**(§7.21「再現不可能な手動生成物」)。
`tick_calibration.json` の画素座標はそのクロップのバイト列に対して
定義されているので、枠が無いと主条件2(人が軸を校正)は原理的に
再現できない。

受け入れ基準は**バイト一致**である。逆算した矩形で元画像を切り直した
結果が、コミット済みクロップと1バイトも違わないことを確認できた
エントリだけに `crop` と `final_sha256` を書く。
近似の枠は書かない — 書けば tick_calibration の全画素が静かにずれる。
figure-fetch-distribution §9 の通り、1件でも落ちたら本設計は撤回であり、
このスクリプトの出力はその判定材料そのものである。

使い方:
    PYTHONPATH=$PWD/src python3 scripts/tools/recover_crop_boxes.py
    PYTHONPATH=$PWD/src python3 scripts/tools/recover_crop_boxes.py --apply
    # リポジトリ外の抽出キャッシュも候補に入れる(data/raw/ は gitignore)
    PYTHONPATH=$PWD/src python3 scripts/tools/recover_crop_boxes.py --search-root data/raw/images

既定は dry-run。`--apply` で registry.json を書き換える。
`convert_ground_truth_to_display_units.py` / `promote_tick_ranges.py` と
同じ流儀で、生の dict を直接書き換えてから adapter の round-trip で
検算する(JSON の差分を純粋な追加だけに保つため)。
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from real_chart_bench.adapter.crop_recovery import (
    CROP_ENCODER,
    NO_EXACT_PLACEMENT,
    CropRecoveryResult,
    recover_crop,
    sha256_file,
)
from real_chart_bench.adapter.verified_pairing_registry import parse_registry, serialize_entry

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
CROPS_DIR = REPO_ROOT / "data/verified_pairs/crops"
IMAGES_DIR = REPO_ROOT / "data/verified_pairs/images"

CROP_PREFIX = "data/verified_pairs/crops/"

# 色集合による候補枝刈り。クロップが元画像の部分矩形なら、クロップに
# 現れる画素値はすべて元画像にも現れる(厳密な必要条件)。FFT 探索は
# 元画像1枚あたり数百ミリ秒〜数秒かかるので、これで候補をほぼ全部落とす。
_COLOR_PROBE_PIXELS = 512


def _display(path: str | None) -> str:
    """報告用にリポジトリ相対へ縮める(外部パスはそのまま出す)。"""
    if path is None:
        return "-"
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path


@dataclass
class EntryOutcome:
    """1エントリの逆算結果(報告用)。"""

    paper_id: str
    figure_id: str
    crop_path: str
    status: str
    detail: str = ""
    result: CropRecoveryResult | None = None


def _decoded(path: Path) -> np.ndarray:
    image = Image.open(path)
    image.load()
    return np.asarray(image)


def _packed_colors(array: np.ndarray) -> np.ndarray:
    """画素値を uint64 の1値に畳んで一意集合にする。"""
    flat = array.reshape(-1, 1) if array.ndim == 2 else array.reshape(-1, array.shape[2])
    packed = np.zeros(flat.shape[0], dtype=np.uint64)
    for channel in range(flat.shape[1]):
        packed = (packed << np.uint64(8)) | flat[:, channel].astype(np.uint64)
    return np.unique(packed)


def _fits(source_shape: tuple[int, ...], crop_shape: tuple[int, ...]) -> bool:
    """どの向きかは問わず、クロップが元画像に収まる余地があるか。"""
    source_height, source_width = source_shape[:2]
    crop_height, crop_width = crop_shape[:2]
    upright = crop_height <= source_height and crop_width <= source_width
    turned = crop_width <= source_height and crop_height <= source_width
    return upright or turned


def _plausible(source: np.ndarray, crop: np.ndarray, crop_colors: np.ndarray) -> bool:
    if source.ndim != crop.ndim:
        return False
    if source.ndim == 3 and source.shape[2] != crop.shape[2]:
        return False
    if not _fits(source.shape, crop.shape):
        return False
    return bool(np.isin(crop_colors, _packed_colors(source)).all())


def _candidate_sources(paper_id: str, crop_path: Path, roots: list[Path], wide: bool) -> list[Path]:
    """候補の元画像。同一論文を先に、次に(--wide なら)全体を見る。"""
    ordered: list[Path] = []

    def add(paths) -> None:
        for path in sorted(paths):
            if path.is_file() and path != crop_path and path not in ordered:
                ordered.append(path)

    for root in roots:
        add((root / paper_id).glob("*"))
    # 5904 の corrected_fig5.png のように、別のクロップから派生したものがある
    add((CROPS_DIR / paper_id).glob("*"))
    if wide:
        for root in roots:
            add(root.glob("*/*"))
        add(CROPS_DIR.glob("*/*"))
    return ordered


def _recover_one(entry: dict, roots: list[Path], wide: bool) -> EntryOutcome:
    relative = entry["image_path"]
    crop_path = REPO_ROOT / relative
    base = EntryOutcome(
        paper_id=entry["paper_id"],
        figure_id=entry["figure_id"],
        crop_path=relative,
        status="",
    )

    if not crop_path.exists():
        base.status = "crop_file_absent"
        base.detail = "クロップ自体がリポジトリに無い(ライセンス制約で削除済み)"
        return base

    crop = _decoded(crop_path)
    crop_colors = _packed_colors(crop)
    candidates = _candidate_sources(entry["paper_id"], crop_path, roots, wide)

    checked = 0
    best: CropRecoveryResult | None = None
    for candidate in candidates:
        try:
            source = _decoded(candidate)
        except OSError:
            continue
        if not _plausible(source, crop, crop_colors):
            continue
        checked += 1
        result = recover_crop(candidate, crop_path)
        if result.is_accepted:
            base.status = "recovered"
            base.result = result
            return base
        if result.failure_reason != NO_EXACT_PLACEMENT and best is None:
            best = result

    if best is not None:
        base.status = best.failure_reason
        base.result = best
        base.detail = f"source={_display(best.source_image_path)}"
        return base

    if not candidates:
        base.status = "source_absent"
        base.detail = "元画像の候補がリポジトリに1件も無い"
    elif checked == 0:
        base.status = "source_absent"
        base.detail = f"候補 {len(candidates)}件はすべてサイズ・色集合で除外(別の図の画像)"
    else:
        base.status = "not_found"
        base.detail = f"候補 {checked}件を厳密探索したが一致なし"
    base.result = CropRecoveryResult(
        crop_path=relative, source_image_path=None, final_sha256=sha256_file(crop_path)
    )
    return base


def _print_report(outcomes: list[EntryOutcome], attempted: int) -> None:
    counts = collections.Counter(o.status for o in outcomes)
    recovered = counts["recovered"]

    print("=" * 78)
    print("第0段 クロップ枠の逆算 — 受け入れ基準はバイト一致")
    print(f"判定に使った符号化手順: {CROP_ENCODER}")
    print("=" * 78)
    print(f"クロップを指すエントリ: {len(outcomes)}")
    print(f"元画像があり探索した   : {attempted}")
    print(f"バイト一致で逆算できた : {recovered} / {attempted}")
    print()
    print("内訳:")
    for status, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {status:30s} {count:4d}")
    print()

    for status in sorted(set(counts) - {"recovered", "source_absent", "crop_file_absent"}):
        print(f"--- {status} ---")
        for outcome in outcomes:
            if outcome.status != status:
                continue
            result = outcome.result
            box = result.box if result is not None else None
            print(
                f"  {outcome.paper_id}/{outcome.figure_id} {outcome.crop_path}"
                f"  box={box}"
                f"  orientation={getattr(result, 'orientation', None)}"
                f"  pixels_identical={getattr(result, 'pixels_identical', None)}"
                f"  {outcome.detail}"
            )
        print()

    print("--- 逆算できた枠 ---")
    for outcome in outcomes:
        if outcome.status != "recovered":
            continue
        recipe = outcome.result.recipe
        print(
            f"  {outcome.paper_id}/{outcome.figure_id} {outcome.crop_path}"
            f"  <- {_display(recipe.source_image_path)} box={list(recipe.box)}"
            f" rotation={recipe.rotation_deg}"
        )
    print()

    absent = [o for o in outcomes if o.status == "source_absent"]
    print(f"--- 元画像がリポジトリに無い: {len(absent)}件 ---")
    by_paper = collections.Counter(o.paper_id for o in absent)
    for paper_id, count in sorted(by_paper.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  paper {paper_id}: {count}件")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="registry.json を実際に書き換える"
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        metavar="DIR",
        help="元画像を探す追加のディレクトリ(<DIR>/<paper_id>/ を見る)。"
        "data/raw/images のような gitignore 下の抽出キャッシュを指定する用",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="同一論文で見つからない場合、全論文の画像も候補にする(取り違え検出)",
    )
    args = parser.parse_args()

    roots = [IMAGES_DIR]
    for raw_root in args.search_root:
        root = Path(raw_root)
        roots.append(root if root.is_absolute() else REPO_ROOT / root)

    registry = json.loads(REGISTRY_PATH.read_text())
    outcomes: list[EntryOutcome] = []
    attempted = 0

    for entry in registry:
        image_path = entry.get("image_path")
        if not image_path or not image_path.startswith(CROP_PREFIX):
            continue
        outcome = _recover_one(entry, roots, args.wide)
        outcomes.append(outcome)
        if outcome.status != "crop_file_absent" and outcome.status != "source_absent":
            attempted += 1
        if outcome.status != "recovered":
            continue

        recipe = outcome.result.recipe
        # 記録する source_image_path はリポジトリ相対に正規化する。
        # リポジトリ外(--search-root で与えた gitignore 下のキャッシュ)の
        # 画像は配布物ではないので、レシピとして書かない。
        try:
            relative_source = Path(recipe.source_image_path).resolve().relative_to(REPO_ROOT)
        except ValueError:
            outcome.status = "source_outside_repo"
            outcome.detail = (
                f"バイト一致したが元画像 {recipe.source_image_path} はリポジトリ外"
                "(配布されないのでレシピとして記録しない)"
            )
            continue
        entry["crop"] = {
            "source_image_path": relative_source.as_posix(),
            "box": [int(value) for value in recipe.box],
            "rotation_deg": recipe.rotation_deg,
        }
        entry["final_sha256"] = outcome.result.final_sha256

    # 手で書き換えた dict が、domain モデルの round-trip と一致することを検算
    for entry in registry:
        serialize_entry(parse_registry([entry])[0], base=entry)

    _print_report(outcomes, attempted)

    if args.apply:
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
        written = sum(1 for o in outcomes if o.status == "recovered")
        print(f"registry.json に {written}件の crop/final_sha256 を書き込んだ")
    else:
        print("dry-run(--apply で registry.json を書き換える)")


if __name__ == "__main__":
    main()
