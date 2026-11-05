"""Inference launcher checks use synthetic executables, not native models."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (("run_inference.py", "run_inference", "llama-cli"),
           ("run_inference_server.py", "run_server", "llama-server"))


def load_launcher(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.args = mock.Mock(model="relative model.gguf", n_predict=8, threads=2,
                            prompt="literal prompt", ctx_size=128, temperature=0.5,
                            conversation=False, host="127.0.0.1", port=8080)
    return module


class LauncherPathTests(unittest.TestCase):
    def check_launcher(self, system, release_exists):
        for filename, entrypoint, binary in SCRIPTS:
            with self.subTest(filename=filename):
                module = load_launcher(filename)
                with tempfile.TemporaryDirectory() as directory:
                    old_cwd = Path.cwd()
                    try:
                        os.chdir(directory)
                        with mock.patch.object(module.platform, "system", return_value=system), \
                             mock.patch.object(module.os.path, "exists", return_value=release_exists), \
                             mock.patch.object(module.subprocess, "run") as run:
                            getattr(module, entrypoint)()
                    finally:
                        os.chdir(old_cwd)
                expected = ROOT / "build/bin"
                if system == "Windows" and release_exists:
                    expected = expected / "Release" / (binary + ".exe")
                else:
                    expected = expected / binary
                self.assertEqual(run.call_args.args[0][0], str(expected))
                self.assertEqual(run.call_args.args[0][1:3], ["-m", "relative model.gguf"],
                                 "User model paths must retain caller-relative semantics")
                self.assertFalse(run.call_args.kwargs["shell"])

    def test_posix_launchers_are_independent_of_working_directory(self):
        self.check_launcher("Linux", False)

    def test_windows_release_path_is_checkout_relative(self):
        self.check_launcher("Windows", True)

    def test_windows_fallback_path_is_checkout_relative(self):
        self.check_launcher("Windows", False)

    @unittest.skipUnless(os.name == "posix", "Synthetic executable fixture uses POSIX permissions")
    def test_cli_launches_synthetic_executable_from_another_directory(self):
        for filename, _, binary in SCRIPTS:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                checkout = Path(directory) / "synthetic checkout"
                workdir = Path(directory) / "caller directory"
                bindir = checkout / "build/bin"
                bindir.mkdir(parents=True)
                workdir.mkdir()
                script = checkout / filename
                shutil.copy2(ROOT / filename, script)
                executable = bindir / binary
                executable.write_text("#!" + sys.executable + "\nimport json, sys\nprint('ARGV=' + json.dumps(sys.argv[1:]))\n", encoding="utf-8")
                executable.chmod(0o755)
                result = subprocess.run([sys.executable, str(script), "-m", "relative model.gguf", "-p", "literal ; $ prompt"],
                                        cwd=workdir, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, "Launcher must find its checkout's executable: " + result.stderr)
                line = next(line for line in result.stdout.splitlines() if line.startswith("ARGV="))
                args = json.loads(line[5:])
                self.assertEqual(args[args.index("-m") + 1], "relative model.gguf")
                self.assertEqual(args[args.index("-p") + 1], "literal ; $ prompt")


if __name__ == "__main__":
    unittest.main()
