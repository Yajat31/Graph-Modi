"""Verify the proposal body fits two pages, excluding references.

Recompiles a copy with PDF compression disabled so page content streams can be
read without a PDF library, then counts text lines that appear on page 3 before
the References heading. Exits non-zero if any body text spills past page 2.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
WORK = HERE / "_pagecheck"


def show_text_lines(stream: bytes) -> list[str]:
    lines = []
    for m in re.finditer(rb"\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj", stream, re.S):
        chunk = m.group(1) or m.group(2) or b""
        parts = re.findall(rb"\((?:\\.|[^()\\])*\)", chunk) or [chunk]
        lines.append(b"".join(p.strip(b"()") for p in parts).decode("latin-1"))
    return lines


def main() -> int:
    WORK.mkdir(exist_ok=True)
    for name in ("acl.sty", "acl_natbib.bst", "custom.bib"):
        shutil.copy(HERE / name, WORK / name)
    tex = (HERE / "final_proposal.tex").read_text().replace(
        "\\begin{document}",
        "\\pdfcompresslevel=0\n\\pdfobjcompresslevel=0\n\\begin{document}",
        1,
    )
    (WORK / "m.tex").write_text(tex)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "m.tex"],
        cwd=WORK,
        capture_output=True,
    )
    pdf = (WORK / "m.pdf").read_bytes()
    streams = sorted(
        (m.start(), m.group(1))
        for m in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.S)
    )
    pages = [ls for ls in (show_text_lines(s) for _, s in streams) if len(ls) > 5]

    print(f"pages: {len(pages)}  lines per page: {[len(p) for p in pages]}")
    if len(pages) < 3:
        print("OK: body and references fit within 2 pages")
        return 0

    third = pages[2]
    ref_at = next((i for i, line in enumerate(third) if "References" in line), None)
    if ref_at is None:
        print(f"FAIL: {len(third)} body lines on page 3, no References heading")
        return 1
    if ref_at == 0:
        print("OK: body ends on page 2; references start on page 3")
        return 0
    print(f"FAIL: {ref_at} body lines spill onto page 3:")
    for line in third[:ref_at]:
        print("   |", line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
