"""Check real ELF architecture selection without running ARM code or DDS."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "deploy/g1/cmake/OnnxRuntime.cmake"
ARM = ROOT / "deploy/g1/third_party/onnxruntime-linux-aarch64-1.23.2"
X86 = ROOT / "deploy/g1/third_party/onnxruntime-linux-x64-1.22.0"


@unittest.skipUnless(shutil.which("cmake"), "CMake required")
class OnnxRuntimeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        (self.directory / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.16)\n'
            'project(ort_selection NONE)\n'
            'set(CMAKE_SYSTEM_NAME Linux)\n'
            'set(CMAKE_SYSTEM_PROCESSOR "${TEST_PROCESSOR}")\n'
            f'include("{MODULE}")\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${ORT_LIBRARY}")\n'
        )

    def configure(self, processor, *options):
        return subprocess.run(
            ["cmake", "-S", str(self.directory), "-B", str(self.directory / "build"),
             f"-DTEST_PROCESSOR={processor}", *options],
            capture_output=True, text=True, timeout=15)

    def selected(self, processor, expected, *options):
        result = self.configure(processor, *options)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        library = Path((self.directory / "build/selected.txt").read_text())
        self.assertEqual(library.resolve(), expected.resolve())

    def test_native_x86_distribution_and_aliases(self):
        for processor in ("x86_64", "AMD64", "x64"):
            with self.subTest(processor=processor):
                self.selected(processor, X86 / "lib/libonnxruntime.so.1.22.0")

    def test_arm_distribution_and_reconfigure_without_stale_x86_cache(self):
        self.selected("x86_64", X86 / "lib/libonnxruntime.so.1.22.0")
        for processor in ("aarch64", "arm64", "ARM64"):
            with self.subTest(processor=processor):
                self.selected(processor, ARM / "lib/libonnxruntime.so.1.23.2")
        cache = (self.directory / "build/CMakeCache.txt").read_text()
        self.assertIn("ONNXRUNTIME_ROOT:PATH=\n", cache)
        self.assertNotIn("ORT_LIBRARY:FILEPATH=", cache)

    def test_custom_root_replaces_stale_library_cache(self):
        custom = self.directory / "custom-runtime"
        custom.symlink_to(X86, target_is_directory=True)
        self.selected("x86_64", custom / "lib/libonnxruntime.so.1.22.0",
                      f"-DONNXRUNTIME_ROOT={custom}",
                      f"-DORT_LIBRARY={ARM}/lib/libonnxruntime.so.1.23.2")

    def test_wrong_architecture_override_is_rejected(self):
        for processor, root in (("aarch64", X86), ("x86_64", ARM)):
            with self.subTest(processor=processor):
                result = self.configure(processor, f"-DONNXRUNTIME_ROOT={root}")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("architecture mismatch", result.stderr)

    def test_unsupported_architecture_is_rejected(self):
        result = self.configure("armv7l")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported ONNX Runtime target architecture", result.stderr)

    def test_incomplete_distribution_is_rejected(self):
        result = self.configure("aarch64", f"-DONNXRUNTIME_ROOT={self.directory}/missing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("headers not found", result.stderr)
        incomplete = self.directory / "incomplete"
        (incomplete / "include").mkdir(parents=True)
        (incomplete / "include/onnxruntime_cxx_api.h").touch()
        result = self.configure("aarch64", f"-DONNXRUNTIME_ROOT={incomplete}")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("library not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
