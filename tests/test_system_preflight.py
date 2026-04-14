import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.system_preflight import run_preflight


class SystemPreflightTests(unittest.TestCase):
    def test_preflight_reuses_cached_success_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core").mkdir()
            (root / "config").mkdir()
            (root / ".runtime").mkdir()
            (root / "core" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "config" / "sample.json").write_text('{"ok": true}\n', encoding="utf-8")
            state_path = root / ".runtime" / "preflight.json"

            with patch("core.system_preflight.SOURCE_GLOBS", ("core/**/*.py",)), patch(
                "core.system_preflight.JSON_GLOBS", ("config/*.json",)
            ), patch("core.system_preflight.CRITICAL_IMPORTS", ()), patch(
                "core.system_preflight.REQUIRED_ARTIFACTS", ()
            ), patch("core.system_preflight.OPTIONAL_ARTIFACTS", ()):
                first = run_preflight(root=root, state_path=state_path, deep_model_checks=False)
                second = run_preflight(root=root, state_path=state_path, deep_model_checks=False)

            self.assertTrue(first.ok)
            self.assertFalse(first.skipped)
            self.assertTrue(second.ok)
            self.assertTrue(second.skipped)

    def test_preflight_blocks_broken_python_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core").mkdir()
            (root / "config").mkdir()
            (root / ".runtime").mkdir()
            (root / "core" / "broken.py").write_text("def nope(:\n    pass\n", encoding="utf-8")
            (root / "config" / "sample.json").write_text('{"ok": true}\n', encoding="utf-8")
            state_path = root / ".runtime" / "preflight.json"

            with patch("core.system_preflight.SOURCE_GLOBS", ("core/**/*.py",)), patch(
                "core.system_preflight.JSON_GLOBS", ("config/*.json",)
            ), patch("core.system_preflight.CRITICAL_IMPORTS", ()), patch(
                "core.system_preflight.REQUIRED_ARTIFACTS", ()
            ), patch("core.system_preflight.OPTIONAL_ARTIFACTS", ()):
                result = run_preflight(root=root, state_path=state_path, deep_model_checks=False)

            self.assertFalse(result.ok)
            self.assertTrue(any(issue.check == "compile" for issue in result.issues))

    def test_preflight_blocks_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "core").mkdir()
            (root / "config").mkdir()
            (root / ".runtime").mkdir()
            (root / "core" / "valid.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "config" / "sample.json").write_text('{"ok": \n', encoding="utf-8")
            state_path = root / ".runtime" / "preflight.json"

            with patch("core.system_preflight.SOURCE_GLOBS", ("core/**/*.py",)), patch(
                "core.system_preflight.JSON_GLOBS", ("config/*.json",)
            ), patch("core.system_preflight.CRITICAL_IMPORTS", ()), patch(
                "core.system_preflight.REQUIRED_ARTIFACTS", ()
            ), patch("core.system_preflight.OPTIONAL_ARTIFACTS", ()):
                result = run_preflight(root=root, state_path=state_path, deep_model_checks=False)

            self.assertFalse(result.ok)
            self.assertTrue(any(issue.check == "json" for issue in result.issues))


if __name__ == "__main__":
    unittest.main()
