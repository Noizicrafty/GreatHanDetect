"""Clone a pinned clyv/smoking-detection revision into vendor/."""
from pathlib import Path
import subprocess
from common import ROOT

UPSTREAM = "https://github.com/clyv/smoking-detection.git"
COMMIT = "e7bba89b0336119363cf4d9f9e31b2e4a67e9233"
p = ROOT / "vendor" / "smoking-detection"
if (p / ".git").exists():
    revision = subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise SystemExit(f"Existing source is {revision}; expected {COMMIT}. Move vendor/smoking-detection aside to re-clone.")
    print(f"Pinned upstream already present: {revision}")
else:
    if p.exists() and any(p.iterdir()):
        raise SystemExit(f"{p} exists and is not empty; move it aside before cloning.")
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", UPSTREAM, str(p)], check=True)
    subprocess.run(["git", "-C", str(p), "checkout", "--detach", COMMIT], check=True)
    if not (p / "smokedet" / "fusion.py").exists():
        raise SystemExit("Unexpected upstream layout; missing smokedet/fusion.py")
    print(f"Original source pinned to {COMMIT}")
