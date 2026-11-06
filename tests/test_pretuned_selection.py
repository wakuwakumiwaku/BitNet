"""Select shipped kernels without compiling or downloading a model."""
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pretuned_setup", ROOT / "setup_env.py")
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)
assert setup.__file__ is not None
assert Path(setup.__file__).resolve() == ROOT / "setup_env.py"


class PretunedTests(unittest.TestCase):
    def exercise(self, arch, quant, pretuned):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "include").mkdir()
            (work / "logs").mkdir()
            (work / "utils").mkdir()
            script = "codegen_tl1.py" if arch == "arm64" else "codegen_tl2.py"
            shutil.copy2(ROOT / "utils" / script, work / "utils" / script)
            shutil.copytree(ROOT / "preset_kernels", work / "preset_kernels")
            setup.args = SimpleNamespace(hf_repo="1bitLLM/bitnet_b1_58-large",
                                         model_dir="unused", quant_type=quant,
                                         use_pretuned=pretuned, log_dir=str(work / "logs"))
            previous = Path.cwd()
            try:
                os.chdir(work)
                # Only the code generation step is real; package installation,
                # native compilation and model preparation cannot execute.
                with mock.patch.object(setup, "system_info", return_value=("Linux", arch)), \
                     mock.patch.object(setup, "setup_gguf"), \
                     mock.patch.object(setup, "compile"), \
                     mock.patch.object(setup, "prepare_model"):
                    setup.main()
            finally:
                os.chdir(previous)
            for log in (work / "logs").iterdir():
                print("CODEGEN LOG", log.name, log.read_text())
            header = (work / "include/bitnet-lut-kernels.h").read_bytes()
            config = (work / "include/kernel_config.ini").read_bytes()
            preset = work / "preset_kernels/bitnet_b1_58-large"
            print("SOURCE", setup.__file__, "ARCH", arch, "PRETUNED", pretuned,
                  "HEADER_SHA256", hashlib.sha256(header).hexdigest())
            if pretuned:
                expected_header = (preset / f"bitnet-lut-kernels-{quant}.h").read_bytes()
                self.assertEqual(hashlib.sha256(header).hexdigest(),
                                 hashlib.sha256(expected_header).hexdigest(),
                                 "Selected pretuned header must not be regenerated")
                self.assertEqual(config, (preset / f"kernel_config_{quant}.ini").read_bytes(),
                                 "Use the matching pretuned tiling configuration")
                self.assertFalse((work / "logs/codegen.log").exists(),
                                 "Pretuned selection must not execute code generation")
            else:
                self.assertTrue((work / "logs/codegen.log").exists())
                self.assertIn(b"Kernels_0", config)
                self.assertIn(b"GGML_BITNET", header)

    def test_arm_pretuned_selection(self):
        self.exercise("arm64", "tl1", True)

    def test_x86_pretuned_selection(self):
        self.exercise("x86_64", "tl2", True)

    def test_default_generation_still_runs_on_both_architectures(self):
        for arch, quant in (("arm64", "tl1"), ("x86_64", "tl2")):
            with self.subTest(arch=arch):
                self.exercise(arch, quant, False)


if __name__ == "__main__":
    unittest.main()
