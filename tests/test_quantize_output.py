"""Quantization failure handling with tiny synthetic executables, not models."""
import importlib.util
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quantize_embeddings", ROOT / "utils/quantize_embeddings.py")
assert SPEC is not None and SPEC.loader is not None
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)
assert Path(quantize.__file__).resolve() == ROOT / "utils/quantize_embeddings.py"


@unittest.skipUnless(os.name == "posix", "Synthetic executable uses POSIX permissions")
class QuantizeOutputTests(unittest.TestCase):
    def fixture(self, root, fail=True):
        source = root / "input.gguf"
        source.write_bytes(b"synthetic input, never a model")
        binary = root / "synthetic quantizer"
        binary.write_text("#!" + sys.executable + "\nimport pathlib, sys\n"
                          "pathlib.Path(sys.argv[4]).write_bytes(" +
                          ("b'partial'" if fail else "b'complete'") + ")\n"
                          + ("sys.exit(9)\n" if fail else ""), encoding="utf-8")
        binary.chmod(0o755)
        return quantize.EmbeddingQuantizer(source, root / "outputs", binary, binary, root / "stats")

    def test_failed_quantizer_does_not_poison_retry_with_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = self.fixture(root)
            output = runner.output_dir / "ggml-model-i2_s-embed-f16.gguf"
            canary = runner.output_dir / "unrelated.gguf"
            canary.write_bytes(b"keep")
            self.assertFalse(runner.quantize("F16", "f16"))
            self.assertFalse(output.exists(), "A failed quantizer must not leave a reusable final model")
            self.assertEqual(canary.read_bytes(), b"keep")
            self.fixture(root, fail=False)
            self.assertTrue(runner.quantize("F16", "f16"))
            self.assertEqual(output.read_bytes(), b"complete", "Retry must run the quantizer, not reuse partial bytes")
            runner.cleanup_model("f16")
            self.assertFalse(output.exists())
            self.assertEqual(list(runner.output_dir.iterdir()), [canary])

    def test_timeout_cleans_only_this_attempts_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.fixture(Path(directory))
            def timeout(command, **kwargs):
                Path(command[4]).write_bytes(b"partial")
                raise subprocess.TimeoutExpired(command, 600)
            with mock.patch.object(quantize.subprocess, "run", side_effect=timeout):
                self.assertFalse(runner.quantize("F16", "f16"))
            self.assertEqual(list(runner.output_dir.iterdir()), [], "Timed-out partial models must not be reused")

    def test_existing_model_is_never_replaced_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.fixture(Path(directory))
            output = runner.output_dir / "ggml-model-i2_s-embed-f16.gguf"
            output.write_bytes(b"existing model")
            with mock.patch.object(quantize.subprocess, "run") as execute:
                self.assertTrue(runner.quantize("F16", "f16"))
                execute.assert_not_called()
            runner.cleanup_model("f16")
            self.assertEqual(output.read_bytes(), b"existing model")


if __name__ == "__main__":
    unittest.main()
