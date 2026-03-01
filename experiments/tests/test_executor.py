"""Tests for sandboxed subprocess execution."""

import time

import pytest

from experiments.evaluation.executor import execute_code


class TestExecuteCode:
    def test_passing_code(self):
        passed, stderr, exit_code = execute_code('print("hello")')
        assert passed is True
        assert stderr == ""
        assert exit_code == 0

    def test_syntax_error(self):
        passed, stderr, exit_code = execute_code("def foo(")
        assert passed is False
        assert "SyntaxError" in stderr
        assert exit_code != 0

    def test_runtime_error_division_by_zero(self):
        passed, stderr, exit_code = execute_code("1/0")
        assert passed is False
        assert "ZeroDivisionError" in stderr
        assert exit_code != 0

    def test_timeout_infinite_loop(self):
        start = time.time()
        passed, stderr, exit_code = execute_code("while True: pass", timeout=2)
        elapsed = time.time() - start
        assert passed is False
        assert stderr == "timeout"
        assert exit_code == -1
        # Should complete within timeout + 1s margin
        assert elapsed < 4.0

    def test_wrong_answer_assertion_error(self):
        passed, stderr, exit_code = execute_code("assert 1 == 2")
        assert passed is False
        assert "AssertionError" in stderr
        assert exit_code != 0

    def test_sys_exit_does_not_crash_executor(self):
        passed, stderr, exit_code = execute_code("import sys; sys.exit(1)")
        assert passed is False
        assert exit_code != 0

    def test_sys_exit_zero_passes(self):
        passed, stderr, exit_code = execute_code("import sys; sys.exit(0)")
        assert passed is True
        assert exit_code == 0

    def test_empty_script(self):
        passed, stderr, exit_code = execute_code("")
        assert passed is False
        assert exit_code != 0

    def test_whitespace_only_script(self):
        passed, stderr, exit_code = execute_code("   \n\n  ")
        assert passed is False
        assert exit_code != 0

    def test_import_error(self):
        passed, stderr, exit_code = execute_code("import nonexistent_module_xyz")
        assert passed is False
        assert "ModuleNotFoundError" in stderr or "ImportError" in stderr
        assert exit_code != 0

    def test_name_error(self):
        passed, stderr, exit_code = execute_code("print(undefined_variable_xyz)")
        assert passed is False
        assert "NameError" in stderr
        assert exit_code != 0

    def test_multiline_passing_code(self):
        script = "def add(a, b):\n    return a + b\nassert add(1, 2) == 3"
        passed, stderr, exit_code = execute_code(script)
        assert passed is True
        assert exit_code == 0
