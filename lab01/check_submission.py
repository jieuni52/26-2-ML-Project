"""Submission checker — validates zip name, structure, and results.json schema.

Usage: python check_submission.py {student_id}_{name}_labXX.zip
DO NOT MODIFY. The same checker is used for every lab this semester.
"""
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

REQUIRED = ["report.pdf", "results.json", "README.md"]
RESULT_KEYS = {"lab", "student_id", "name", "seed", "metrics", "runtime_seconds"}
# 이름은 한글 또는 로마자만. 공백·숫자·특수문자는 받지 않습니다.
NAME_RE = re.compile(r"^(\d{8})_([A-Za-z가-힣]+)_lab(\d{2}(?:_\d)?)\.zip$")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    fail.count += 1
fail.count = 0


def ok(msg: str) -> None:
    print(f"  [ ok ] {msg}")


def note(msg: str) -> None:
    """Advisory only — does not block submission."""
    print(f"  [note] {msg}")


def main(path_str: str) -> int:
    path = Path(path_str)
    # 맥에서 만든 한글 파일명은 자모가 분해된(NFD) 형태입니다. 검사 전에 NFC로 맞춥니다.
    # 파일을 열 때는 원래 path 를 그대로 씁니다.
    name = unicodedata.normalize("NFC", path.name)
    print(f"Checking {name} ...")
    m = NAME_RE.match(name)
    if not m:
        fail("zip name must be {8-digit student_id}_{name}_labXX.zip — "
             "the name in Korean or roman letters, with no spaces, digits, or symbols "
             "(e.g. 20261234_홍길동_lab01.zip or 20261234_HongGildong_lab01.zip)")
    else:
        ok(f"zip name (student_id={m.group(1)}, lab{m.group(3)})")
    if not path.exists():
        fail("file not found"); return 1
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        # tolerate a single top-level folder inside the zip
        roots = {n.split("/")[0] for n in names if "/" in n}
        prefix = f"{roots.pop()}/" if len(roots) == 1 and not any("/" not in n for n in names) else ""
        stripped = [n[len(prefix):] for n in names if n != prefix]

        if not any(n.startswith("src/") and n.endswith(".py") for n in stripped):
            fail("src/ must contain at least one .py file (your logic — notebooks are not graded)")
        else:
            ok("src/ contains .py code")
        for req in REQUIRED:
            if req in stripped:
                ok(req)
            else:
                fail(f"missing {req} at zip top level")
        junk = [n for n in stripped if "__pycache__" in n or n.endswith(".DS_Store") or ".ipynb_checkpoints" in n]
        if junk:
            fail(f"remove junk files: {junk[:3]} ...")
        nbs = [n for n in stripped if n.endswith(".ipynb")]
        if nbs:
            note("notebooks are not part of the submission and will be ignored — "
                 "your figures belong in report.pdf")
        csvs = [n for n in stripped if n.endswith((".csv", ".tsv"))]
        if csvs:
            note("datasets are not submitted; remove them to keep the zip small")
        big = [i.filename for i in z.infolist() if i.file_size > 10 * 2**20]
        if big:
            fail(f"files over 10 MB not allowed: {big}")
        if "results.json" in stripped:
            try:
                r = json.loads(z.read(prefix + "results.json"))
                missing = RESULT_KEYS - set(r)
                if missing:
                    fail(f"results.json missing keys: {sorted(missing)}")
                elif m and r.get("student_id") != m.group(1):
                    fail("results.json student_id does not match zip name")
                elif r.get("seed") != 42:
                    fail("seed must be 42")
                else:
                    ok("results.json schema")
            except json.JSONDecodeError:
                fail("results.json is not valid JSON")
    if fail.count == 0:
        print("PASSED — ready to submit.")
        return 0
    print(f"{fail.count} problem(s) — fix before submitting.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1]))
