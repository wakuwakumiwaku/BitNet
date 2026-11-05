"""CPU-only checks for benchmark wrapper success and failure propagation."""
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e2e_benchmark", ROOT / "utils/e2e_benchmark.py")
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkExitTests(unittest.TestCase):
    def test_successful_command_returns_normally(self):
        with mock.patch.object(benchmark.subprocess, "run") as run:
            try:
                benchmark.run_command(["synthetic-bench", "model with spaces"])
            except SystemExit as exc:
                self.fail(f"Successful benchmark must not exit with {exc.code}")
        run.assert_called_once_with(["synthetic-bench", "model with spaces"], shell=False, check=True)

    def test_failed_command_exits_nonzero(self):
        error = subprocess.CalledProcessError(7, ["synthetic-bench"])
        with mock.patch.object(benchmark.subprocess, "run", side_effect=error):
            with self.assertLogs(level="ERROR"):
                with self.assertRaises(SystemExit) as caught:
                    benchmark.run_command(["synthetic-bench"])
        self.assertEqual(caught.exception.code, 1)

    def test_cli_success_and_failure(self):
        for returncode in (0, 7):
            with self.subTest(returncode=returncode):
                error = subprocess.CalledProcessError(returncode, ["synthetic-bench"]) if returncode else None
                with mock.patch.object(sys, "argv", ["e2e_benchmark.py", "-m", "synthetic model.gguf"]), \
                     mock.patch("os.path.exists", return_value=True), \
                     mock.patch("subprocess.run", side_effect=error) as run:
                    exit_code = 0
                    try:
                        runpy.run_path(str(ROOT / "utils/e2e_benchmark.py"), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = exc.code
                self.assertEqual(exit_code, 1 if returncode else 0)
                self.assertEqual(run.call_args.args[0][1:3], ["-m", "synthetic model.gguf"])

    def test_logged_command_keeps_existing_success_and_failure_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(benchmark, "args", mock.Mock(log_dir=tmp), create=True):
                for fails in (False, True):
                    with self.subTest(fails=fails):
                        error = subprocess.CalledProcessError(7, ["synthetic-bench"]) if fails else None
                        with mock.patch.object(benchmark.subprocess, "run", side_effect=error):
                            if fails:
                                with self.assertLogs(level="ERROR"), self.assertRaises(SystemExit):
                                    benchmark.run_command(["synthetic-bench"], log_step="benchmark")
                            else:
                                benchmark.run_command(["synthetic-bench"], log_step="benchmark")
                self.assertTrue((Path(tmp) / "benchmark.log").exists())


if __name__ == "__main__":
    unittest.main()
