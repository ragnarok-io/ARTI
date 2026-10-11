"""Exercise the exact public wheel in a consumer without the repository source tree.

The wheel is extracted into an isolated import root and reuses the current
interpreter's dependencies. This check does not install dependencies, fetch
models or execute provider services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "test_task_project.py", "test_task_project_cli.py", "test_task_status.py",
    "test_task_scaffold.py", "test_file_lifecycle_workflow.py",
)
MODULES = (
    "task_project.py", "task_scaffold.py", "task_status.py", "recipe_assets.py",
    "recipe_files.py", "recipe_prediction.py", "recipe_episode.py", "recipe_suite.py",
    "recipe_runtime.py", "recipe_export.py", "cli.py", "__init__.py",
    "torch/__init__.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".tmp/installed-task-workflow.json")
    args = parser.parse_args()
    wheels = [args.wheel.resolve()] if args.wheel else sorted((ROOT / "dist").glob("arti_fit-*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file():
        raise SystemExit("expected exactly one built public wheel; run check_package.py first")
    wheel = wheels[0]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    junit = output.with_suffix(".xml")
    with tempfile.TemporaryDirectory(prefix="arti-installed-task-") as temporary:
        scratch = Path(temporary)
        installed = scratch / "wheel"
        consumer = scratch / "consumer"
        selected_wheel = scratch / wheel.name
        shutil.copyfile(wheel, selected_wheel)
        wheel_sha256 = hashlib.sha256(selected_wheel.read_bytes()).hexdigest()
        installed.mkdir()
        consumer.mkdir()
        with zipfile.ZipFile(selected_wheel) as archive:
            archive.extractall(installed)
        hashes = {}
        for name in MODULES:
            source = (ROOT / "src/arti" / name).read_bytes()
            packaged = (installed / "arti" / name).read_bytes()
            if source != packaged:
                raise SystemExit(f"wheel differs from reviewed source: arti/{name}")
            hashes[name] = hashlib.sha256(packaged).hexdigest()
        # Test helpers are the reviewed public fixtures; framework source is absent.
        shutil.copytree(ROOT / "tests", consumer / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "examples", consumer / "examples", ignore=shutil.ignore_patterns("__pycache__"))
        (consumer / "pytest.ini").write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
        (consumer / "installed_assertions.py").write_text(
            "from pathlib import Path\n"
            "import os\n"
            "import arti\n"
            "expected = Path(os.environ['ARTI_WHEEL_ROOT']) / 'arti'\n"
            "assert Path(arti.__file__).resolve().parent == expected.resolve(), "
            "'framework must load from the isolated wheel'\n"
            "Path(os.environ['ARTI_WHEEL_ORIGIN_RECEIPT']).write_text('verified', encoding='utf-8')\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": os.pathsep.join((str(installed), str(consumer))),
            "ARTI_WHEEL_ROOT": str(installed), "PYTHONDONTWRITEBYTECODE": "1",
            "ARTI_WHEEL_ORIGIN_RECEIPT": str(scratch / "wheel-origin.txt"),
            "PYTEST_ADDOPTS": "", "CUDA_VISIBLE_DEVICES": "-1",
            "ACCELERATE_USE_CPU": "true", "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        command = [
            sys.executable, "-m", "pytest", "-c", str(consumer / "pytest.ini"),
            "-p", "installed_assertions", "--rootdir", str(consumer),
            "--basetemp", str(scratch / "pytest-temp"), "--junitxml", str(junit),
            *[str(consumer / "tests" / name) for name in TESTS],
        ]
        result = subprocess.run(command, cwd=consumer, env=env, check=False)
        suites = list(ET.parse(junit).getroot().iter("testsuite")) if junit.is_file() else []
        counts = {
            key: sum(int(suite.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        passed = result.returncode == 0 and counts["tests"] > 0 and all(
            counts[key] == 0 for key in ("failures", "errors", "skipped")
        )
        receipt = {
            "ok": passed, "kind": "installed-public-task-workflow",
            "wheel": wheel.name, "sha256": wheel_sha256,
            "python": ".".join(map(str, sys.version_info[:3])), "platform": sys.platform,
            "method": "isolated wheel extraction with existing interpreter dependencies",
            "dependencies_installed": False, "models_downloaded": False,
            "consumer_contains_framework_source": False,
            "repository_framework_source_on_pythonpath": False,
            "framework_origin_asserted": (scratch / "wheel-origin.txt").is_file(),
            "source_wheel_module_hashes": hashes,
            "test_files": list(TESTS), "counts": counts, "pytest_returncode": result.returncode,
        }
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": passed, "wheel": wheel.name, "counts": counts}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
