"""CHART-Infographics の配布 ZIP を開き、注釈にデータ値が入っている図を数える。

使い方:
    python3 scripts/eval/audit_chartinfo_annotations.py ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0.zip

知りたいのは「注釈ファイルが実際にデータを持っているか」である。配布物は全図に
JSON が付いているので、ファイルが存在することと中身が揃っていることは別になる。
ZIP のまま読むので展開は要らない(1GB 超のため)。

実測の記録は docs/experiments/2026-10-06-chartinfo-dataset-audit.md。
"""

from __future__ import annotations

import collections
import json
import re
import statistics as st
import sys
import zipfile


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    z = zipfile.ZipFile(sys.argv[1])

    tot = collections.Counter()
    with_values = collections.Counter()
    empty = collections.Counter()
    no_key = collections.Counter()
    numeric_x = collections.Counter()
    categorical_x = collections.Counter()
    points = collections.defaultdict(list)
    unreadable: list[tuple[str, str]] = []

    for name in z.namelist():
        if not (name.endswith(".json") and "/annotations_JSON/" in name):
            continue
        chart_type = name.split("/annotations_JSON/")[1].split("/")[0]
        tot[chart_type] += 1
        try:
            doc = json.loads(z.read(name))
        except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
            unreadable.append((name, str(exc)[:60]))
            continue
        # task6 is null for charts whose annotation never reached the data stage
        output = (doc.get("task6") or {}).get("output") or {}
        if "data series" not in output:
            no_key[chart_type] += 1
            continue
        series = output["data series"]
        pts = [p for s in series if isinstance(s, dict) for p in (s.get("data") or [])]
        if not pts:
            empty[chart_type] += 1
            continue
        with_values[chart_type] += 1
        points[chart_type].append(len(pts))
        # a categorical x axis (bar charts) cannot be scored the way we score numeric axes
        bucket = numeric_x if isinstance(pts[0].get("x"), (int, float)) else categorical_x
        bucket[chart_type] += 1

    print(
        f"{'chart type':<22}{'total':>7}{'with data':>11}{'empty':>7}"
        f"{'no key':>8}{'x numeric':>11}{'x text':>8}{'median pts':>12}"
    )
    for ct, n in tot.most_common():
        median = f"{st.median(points[ct]):.0f}" if points[ct] else "-"
        print(
            f"{ct:<22}{n:>7}{with_values[ct]:>11}{empty[ct]:>7}"
            f"{no_key[ct]:>8}{numeric_x[ct]:>11}{categorical_x[ct]:>8}{median:>12}"
        )
    total = sum(tot.values())
    have = sum(with_values.values())
    print(
        f"\n{'TOTAL':<22}{total:>7}{have:>11}{sum(empty.values()):>7}"
        f"{sum(no_key.values()):>8}{sum(numeric_x.values()):>11}"
        f"{sum(categorical_x.values()):>8}"
    )
    print(f"\nデータ値あり: {have}/{total} = {100 * have / total:.1f}%")
    if unreadable:
        print(f"\n読めなかった JSON: {len(unreadable)}")
        for name, exc in unreadable[:5]:
            print(f"  {name}: {exc}")

    # The XML carries VERIFIED_* flags recording how far each image's annotation got,
    # which is the direct evidence for *why* the JSON coverage looks the way it does.
    flags: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    xml_tot = collections.Counter()
    for name in z.namelist():
        if not (name.endswith(".xml") and "/annotations_XML/" in name):
            continue
        ct = name.split("/annotations_XML/")[1].split("/")[0]
        xml_tot[ct] += 1
        for flag in re.findall(r"<(VERIFIED_[\w]+)", z.read(name).decode("utf-8", "replace")):
            flags[ct][flag] += 1
    if xml_tot:
        all_flags = sorted({f for c in flags.values() for f in c})
        print("\n注釈がどこまで進んだか(XML の検証フラグ):")
        print(f"{'chart type':<22}{'total':>7}" + "".join(
            f"{f.replace('VERIFIED_', ''):>11}" for f in all_flags))
        for ct, n in xml_tot.most_common():
            print(f"{ct:<22}{n:>7}" + "".join(f"{flags[ct][f]:>11}" for f in all_flags))


if __name__ == "__main__":
    main()
