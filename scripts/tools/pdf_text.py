"""PDF から本文のテキストを取り出す(zlib + テキスト演算子)。

このマシンには `pdftotext` も `pypdf` も poppler-utils も入っておらず、権限の都合で
入れられない。WebFetch も PDF は解析できない。文献調査で PDF しか公開されていない
論文を読むために、最小限の抽出だけ自前で持つ。

2026-10-08 にこれで4本読めた: FigureSeer(ECCV 2016)、CHART-Info 2024、
PaperUnPlot(TaDA 2026)、CHART-Infographics の指標定義。

**万能ではない。** 文字が埋め込まれていれば取れるが、走査画像だけの PDF では何も出ない。
画像データが本文に混ざることもあるので、読むときは検索語で前後を切り出すのが早い。
合字と括弧は 8進エスケープで出るため置換している。

使い方:
    python3 scripts/tools/pdf_text.py <file.pdf>            # 全文
    python3 scripts/tools/pdf_text.py <file.pdf> <検索語>   # 前後を切り出す
"""

from __future__ import annotations

import pathlib
import re
import sys
import zlib

STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
# PDF のテキスト演算子。Td / TD / T* / ET は改行として扱う。
TOKEN = re.compile(rb"\((?:\\.|[^\\()])*\)|\bTJ\b|\bTj\b|\bTd\b|\bTD\b|\bT\*\b|\bET\b")
NEWLINE_OPERATORS = (b"Td", b"TD", b"T*", b"ET")
# 合字と括弧は 8進エスケープのまま出てくる
SUBSTITUTIONS = ((r"\050", "("), (r"\051", ")"), (r"\013", "ff"), (r"\014", "fi"), (r"\000", "-"))
CONTEXT = 400


def extract(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    chunks: list[str] = []
    for match in STREAM.finditer(raw):
        try:
            stream = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        if b"Tj" not in stream and b"TJ" not in stream:
            continue
        text: list[str] = []
        for token in TOKEN.finditer(stream):
            value = token.group(0)
            if value.startswith(b"("):
                body = re.sub(rb"\\([()\\])", rb"\1", value[1:-1])
                body = body.replace(b"\\n", b" ").replace(b"\\r", b" ").replace(b"\\t", b" ")
                text.append(body.decode("latin-1"))
            elif value in NEWLINE_OPERATORS:
                text.append("\n")
        chunks.append("".join(text))
    joined = "\n".join(chunks)
    for old, new in SUBSTITUTIONS:
        joined = joined.replace(old, new)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", joined))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    text = extract(pathlib.Path(sys.argv[1]))
    if len(sys.argv) < 3:
        print(text)
        return
    # 本文は改行が当てにならないので、連結してから前後を切り出す
    flat = text.replace("\n", "")
    needle = sys.argv[2]
    hits = list(re.finditer(re.escape(needle), flat))
    print(f"{needle!r}: {len(hits)} 件")
    for hit in hits:
        window = flat[max(0, hit.start() - CONTEXT) : hit.end() + CONTEXT]
        # 画像データが混ざることがあるので、印字できない並びは潰す
        print("\n" + re.sub(r"[^\x20-\x7e]{4,}", " ", window))


if __name__ == "__main__":
    main()
