"""Check installed packages for forbidden copyleft licenses.

Allows packages whose PyPI metadata contains GPL classifiers but whose runtime
is known to be under a permissive license (e.g. docutils).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FORBIDDEN = ("GPL", "AGPL", "SSPL")
ALLOWLIST = {
    "docutils",  # runtime is public-domain/BSD; only Emacs helper is GPL
}


def main() -> int:
    output = Path("licenses.json")
    subprocess.run(
        ["pip-licenses", "--format=json", f"--output-file={output}"],
        check=True,
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    bad = [
        (p["Name"], p["License"])
        for p in data
        if p["Name"] not in ALLOWLIST and any(x in p["License"].upper() for x in FORBIDDEN)
    ]
    if bad:
        print("Forbidden licenses found:")
        for name, lic in bad:
            print(f"  {name}: {lic}")
        return 1
    print("No forbidden licenses found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
