"""Exercise the documented conversion CLI with inert owned leaf executables."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "utils/convert-helper-bitnet.py"


@unittest.skipUnless(os.name == "posix", "Synthetic executable fixture uses POSIX permissions")
class ConversionStatusTests(unittest.TestCase):
    def invoke(self, failure):
        with tempfile.TemporaryDirectory(prefix="bitnet-conversion-") as directory:
            root = Path(directory)
            checkout = root / "checkout"
            utils = checkout / "utils"
            utils.mkdir(parents=True)
            bindir = checkout / "build/bin"
            bindir.mkdir(parents=True)
            script = utils / SOURCE.name
            shutil.copy2(SOURCE, script)
            self.assertEqual(script.read_bytes(), SOURCE.read_bytes())
            # Only the helper is copied. No real conversion module or native
            # quantizer is reachable in this synthetic checkout.
            leaf = (
                "import json, os, pathlib, sys\n"
                "stage = pathlib.Path(__file__).name\n"
                "with open(os.environ['STAGE_LOG'], 'a') as f: f.write(stage + '\\n')\n"
                "if stage == 'preprocess-huggingface-bitnet.py':\n"
                "    src = pathlib.Path(sys.argv[sys.argv.index('--input') + 1])\n"
                "    dst = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "    dst.write_bytes(b'synthetic preprocessing output; not a model')\n"
                "elif stage == 'convert-ms-to-gguf-bitnet.py':\n"
                "    pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1]).write_bytes(b'synthetic f32')\n"
                "else:\n"
                "    pathlib.Path(sys.argv[2]).write_bytes(b'synthetic quantized output')\n"
                "if stage == os.environ['FAIL_STAGE']:\n"
                "    print('injected conversion failure', file=sys.stderr)\n"
                "    sys.exit(23)\n"
            )
            for name in ("preprocess-huggingface-bitnet.py", "convert-ms-to-gguf-bitnet.py"):
                (utils / name).write_text(leaf, encoding="utf-8")
            quantizer = bindir / "llama-quantize"
            quantizer.write_text("#!" + sys.executable + "\n" + leaf, encoding="utf-8")
            quantizer.chmod(0o700)
            model = root / "synthetic input with spaces"
            model.mkdir()
            original = model / "model.safetensors"
            original.write_bytes(b"synthetic checkpoint bytes; not a real model")
            canary = model / "unrelated.txt"
            canary.write_bytes(b"untouched")
            log = root / "stages.txt"
            env = dict(os.environ, FAIL_STAGE=failure, STAGE_LOG=str(log))
            argv = [sys.executable, "-B", str(script), str(model)]
            result = subprocess.run(argv, cwd=root, env=env, capture_output=True, timeout=15)
            if os.environ.get("EXTRA_PROBE_RECEIPTS"):
                receipt = Path(os.environ["EXTRA_PROBE_RECEIPTS"]) / (failure or "success")
                receipt.mkdir(parents=True, exist_ok=False)
                (receipt / "stdout.bin").write_bytes(result.stdout)
                (receipt / "stderr.bin").write_bytes(result.stderr)
                (receipt / "receipt.json").write_text(json.dumps({
                    "argv": argv, "cwd": str(root), "explicit_env": env,
                    "inner_exit_code": result.returncode, "timeout_seconds": 15,
                    "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    "stages": log.read_text().splitlines(),
                }, indent=2), encoding="utf-8")
            self.assertEqual(original.read_bytes(), b"synthetic checkpoint bytes; not a real model")
            self.assertFalse((model / "model.safetensors.backup").exists())
            self.assertEqual(canary.read_bytes(), b"untouched")
            return result, log.read_text().splitlines()

    def test_each_failed_stage_exits_nonzero_after_restoring_original(self):
        stages = ["preprocess-huggingface-bitnet.py", "convert-ms-to-gguf-bitnet.py", "llama-quantize"]
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                result, visited = self.invoke(stage)
                self.assertEqual(visited, stages[:index + 1], "Later stages must not run after a failure")
                self.assertIn(b"injected conversion failure", result.stderr)
                self.assertIn(b"An error occurred:", result.stdout)
                self.assertNotIn(b"Convert successfully.", result.stdout)
                self.assertNotEqual(result.returncode, 0, "A failed conversion must not report CLI success")

    def test_successful_pipeline_keeps_zero_and_restores_original(self):
        result, stages = self.invoke("")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(stages), 3)
        self.assertIn(b"Convert successfully.", result.stdout)
        self.assertEqual(result.stderr, b"")


if __name__ == "__main__":
    unittest.main()
