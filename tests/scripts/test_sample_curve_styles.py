"""scripts/eval/sample_curve_styles.py が、正解曲線の色・線種・マーカー形を
正しく読み取れるかのテスト。

マーカー形と線種の単体テストはこのテストの中で作る小さな合成画像だけを使う
(大きな図の画像には依存しない)。実際の図を使う精度テストは1つだけ
(test_accuracy_against_hand_labelled_fixture)で、tests/fixtures/
curve_style_ground_truth.json に目で読んで記録した正解と突き合わせる。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "sample_curve_styles.py"
REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "curve_style_ground_truth.json"

BG = (255, 255, 255)
RED = (220, 30, 30)
BLUE = (30, 30, 220)


@pytest.fixture(scope="module")
def scs():
    spec = importlib.util.spec_from_file_location("sample_curve_styles", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canvas(size=(64, 64), bg=BG) -> Image.Image:
    return Image.new("RGB", size, bg)


def _array(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


# ---------------------------------------------------------------------------
# marker_shape: 基本の形
# ---------------------------------------------------------------------------


def test_marker_shape_detects_filled_square(scs):
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.rectangle((26, 26, 38, 38), fill=RED)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "square"


def test_marker_shape_detects_filled_circle(scs):
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.ellipse((25, 25, 39, 39), fill=RED)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "circle"


def test_marker_shape_crisp_diamond_is_a_known_weak_spot(scs):
    """菱形と丸の塗り割合は、理屈の上ではほぼ同じ(どちらも外接箱の半分)。

    実図の菱形が丸と見分けられるのは、対角線の縁がアンチエイリアス/JPEG で
    にじんで色判定からこぼれ落ち、塗りの割合が理屈の0.5より下がるため
    (tests/fixtures/curve_style_ground_truth.json の実測でも、実図の菱形は
    0.3前後まで下がっている)。にじみのない、くっきりした合成の菱形では
    その効果が起きないので、このヒューリスティックは今のところ丸と読む。
    これは既知の弱点であり強みではない — 将来この判定を直したらこのテストを
    更新すること。回帰ガードとして「少なくとも square や三角形には化けない」
    ことだけは見ておく。
    """
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.polygon([(32, 23), (41, 32), (32, 41), (23, 32)], fill=RED)
    shape = scs.marker_shape(_array(img), 32, 32, RED)
    assert shape == "circle"  # known limitation, not a design goal
    assert shape not in {"square", "triangle-up", "triangle-down"}


def test_marker_shape_detects_triangle_up(scs):
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.polygon([(32, 22), (42, 40), (22, 40)], fill=RED)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "triangle-up"


def test_marker_shape_detects_triangle_down(scs):
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.polygon([(32, 42), (42, 24), (22, 24)], fill=RED)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "triangle-down"


def test_marker_shape_too_small_is_unknown(scs):
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.point((32, 32), fill=RED)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "unknown"


def test_marker_shape_background_only_is_unknown(scs):
    img = _canvas()
    assert scs.marker_shape(_array(img), 32, 32, RED) == "unknown"


# ---------------------------------------------------------------------------
# マーカーの孤立化(連結成分・開き処理・穴埋め)の回帰テスト
# ---------------------------------------------------------------------------


def test_marker_shape_hollow_marker_normalizes_like_solid(scs):
    """中空(縁取りだけ)の四角も、塗りつぶしの四角と同じ形に読めること。

    穴埋め(binary_fill_holes)がないと、縁取りだけのマーカーは外接箱に
    対する塗りの割合が低く出て、四角のはずが丸や菱形に化ける。
    """
    img = _canvas()
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 24, 40, 40), outline=RED, width=2)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "square"


def test_marker_shape_ignores_a_separate_same_colour_neighbour(scs):
    """同じ色でも繋がっていない隣のマーカーは、形の判定に混ざらないこと。

    21283・15452 のような、同じ色の曲線や文字が近くにある密な図で、窓の中の
    色が合う画素を無差別に使うと隣のマーカーが混ざって外接箱が壊れる。
    連結成分による孤立化がないと、四角と三角という別の形が1つの塊に
    合わさってしまい、どちらも正しく読めない。
    """
    img = _canvas((80, 80))
    draw = ImageDraw.Draw(img)
    # 問い合わせる四角
    draw.rectangle((26, 26, 38, 38), fill=RED)
    # 近く(窓には入るが、繋がってはいない)の同色の三角
    draw.polygon([(60, 18), (68, 34), (52, 34)], fill=RED)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "square"
    assert scs.marker_shape(_array(img), 60, 27, RED) == "triangle-up"


def test_marker_shape_strips_a_thin_connecting_line_stub(scs):
    """マーカーに繋がる細い線は、形の判定から取り除かれること。

    線がマーカーと同じ色で直接繋がっていると、連結成分だけでは線の分だけ
    外接箱が伸びて塗りの割合が崩れる。開き処理(収縮して膨張)で線の太さ
    (1〜2px)だけの細い部分を削り、マーカー本体の塊だけを残す。
    """
    img = _canvas((80, 80))
    draw = ImageDraw.Draw(img)
    draw.rectangle((26, 26, 38, 38), fill=RED)
    # 右へ太さ1pxの線を長く伸ばす(別のマーカーに向かう線を模す)
    draw.line((38, 32, 70, 32), fill=RED, width=1)
    assert scs.marker_shape(_array(img), 32, 32, RED) == "square"


# ---------------------------------------------------------------------------
# majority_marker: 同率のときの再現性
# ---------------------------------------------------------------------------


def test_majority_marker_picks_the_clear_majority(scs):
    assert scs.majority_marker(["circle", "circle", "square"]) == "circle"


def test_majority_marker_breaks_ties_by_first_seen(scs):
    assert scs.majority_marker(["circle", "square", "circle", "square"]) == "circle"
    assert scs.majority_marker(["square", "circle", "square", "circle"]) == "square"


def test_majority_marker_ignores_unknowns(scs):
    assert scs.majority_marker(["unknown", "circle", "unknown"]) == "circle"
    assert scs.majority_marker(["unknown", "unknown"]) == "unknown"
    assert scs.majority_marker([]) == "unknown"


# ---------------------------------------------------------------------------
# 線種(segment_duty / line_style)
# ---------------------------------------------------------------------------


def test_line_style_solid_line_reads_solid(scs):
    img = _canvas((80, 80))
    draw = ImageDraw.Draw(img)
    draw.line((10, 40, 70, 40), fill=BLUE, width=2)
    duty = scs.segment_duty(_array(img), (10, 40), (70, 40), BLUE)
    assert duty is not None
    assert scs.line_style([duty]) == "solid"


def test_line_style_dashed_line_reads_dashed(scs):
    img = _canvas((80, 80))
    draw = ImageDraw.Draw(img)
    x = 10
    while x < 70:
        draw.line((x, 40, x + 8, 40), fill=BLUE, width=2)
        x += 16  # 8px 描いて 8px 空ける
    duty = scs.segment_duty(_array(img), (10, 40), (70, 40), BLUE)
    assert duty is not None
    assert scs.line_style([duty]) == "dashed"


def test_line_style_no_line_reads_none(scs):
    img = _canvas((80, 80))
    draw = ImageDraw.Draw(img)
    # 線なし、両端にマーカーだけ(散布図を模す)
    draw.ellipse((6, 36, 14, 44), fill=BLUE)
    draw.ellipse((66, 36, 74, 44), fill=BLUE)
    duty = scs.segment_duty(_array(img), (10, 40), (70, 40), BLUE)
    assert duty is not None
    assert scs.line_style([duty]) == "none"


def test_line_style_empty_duties_is_unknown(scs):
    assert scs.line_style([]) == "unknown"


# ---------------------------------------------------------------------------
# 実図ベースの回帰テスト(唯一、大きな画像に依存するテスト)
# ---------------------------------------------------------------------------


def test_accuracy_against_hand_labelled_fixture(scs):
    """tests/fixtures/curve_style_ground_truth.json (12図57曲線、目視で記録)に
    対する正解率が、このコミットで測った水準を下回らないこと。

    回帰ガード。将来しきい値をいじって下がったらこのテストが落ちる。
    しきい値は 2026-10 時点の実測(全57曲線: マーカー 42/57 = 73.7%、
    ボキャブラリ内53曲線: 42/53 = 79.2%、線種 54/57 = 94.7%)より少し
    低く取ってある(図の読み込み結果が環境でわずかに揺れても壊れないように)。
    星・五角形は現在のマーカー語彙(square/circle/triangle-up/triangle-down/
    diamond)にないので、2つの集計(全曲線 / 語彙内の曲線)の両方を見る。
    """
    fixture = json.loads(FIXTURE.read_text())
    entries = scs.load_entries()
    ground_truth = json.loads((REPO / "data/verified_pairs/ground_truth.json").read_text())
    vocab = {"square", "circle", "triangle-up", "triangle-down", "diamond"}

    marker_total = marker_correct = 0
    vocab_total = vocab_correct = 0
    line_total = line_correct = 0

    for fig in fixture["figures"]:
        fid = fig["figure_id"]
        entry = entries[fid]
        curves = [c for c in ground_truth[fid] if c.get("x")]
        results = scs.styles_for_figure(entry, curves)
        for truth, got in zip(fig["curves"], results, strict=True):
            marker_total += 1
            marker_ok = got["marker"] == truth["marker"]
            marker_correct += marker_ok
            if truth["marker"] in vocab:
                vocab_total += 1
                vocab_correct += marker_ok
            line_total += 1
            line_correct += got["style"] == truth["line"]

    assert marker_total == 57
    assert marker_correct / marker_total >= 0.65, (
        f"marker accuracy (all) dropped to {marker_correct}/{marker_total}"
    )
    assert vocab_correct / vocab_total >= 0.72, (
        f"marker accuracy (vocab-only) dropped to {vocab_correct}/{vocab_total}"
    )
    assert line_correct / line_total >= 0.90, (
        f"line-style accuracy dropped to {line_correct}/{line_total}"
    )
