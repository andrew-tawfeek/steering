from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

from steering import __version__


class PackagingTests(unittest.TestCase):
    def test_pyproject_console_script_and_version(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["dynamic"], ["version"])
        self.assertEqual(pyproject["project"]["scripts"]["steer"], "steering.cli:main")
        self.assertEqual(pyproject["tool"]["setuptools"]["dynamic"]["version"], {"attr": "steering.__version__"})
        self.assertRegex(__version__, r"^\d+\.\d+\.\d+$")

    def test_requirements_match_pyproject_dependencies(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        pyproject_deps = set(pyproject["project"]["dependencies"])
        requirements = {
            line.strip()
            for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertEqual(pyproject_deps, requirements)


if __name__ == "__main__":
    unittest.main()
