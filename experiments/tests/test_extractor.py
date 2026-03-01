"""Tests for code extraction from model outputs."""

from experiments.evaluation.extractor import check_compliance, extract_code


class TestExtractCode:
    def test_extract_from_python_markdown_block(self):
        text = '```python\ndef foo():\n    return 1\n```'
        code, clean = extract_code(text)
        assert clean is True
        assert "def foo():" in code
        assert "return 1" in code

    def test_extract_from_bare_markdown_block(self):
        text = '```\ndef foo():\n    return 1\n```'
        code, clean = extract_code(text)
        assert clean is True
        assert "def foo():" in code
        assert "return 1" in code

    def test_extract_bare_code_no_markdown(self):
        text = "def foo():\n    return 1"
        code, clean = extract_code(text)
        assert clean is True
        assert "def foo():" in code
        assert "return 1" in code

    def test_explanation_before_code_block(self):
        text = 'Here is the solution:\n\n```python\ndef foo():\n    return 1\n```'
        code, clean = extract_code(text)
        assert clean is True
        assert "def foo():" in code
        assert "Here is" not in code

    def test_explanation_after_code_block(self):
        text = '```python\ndef foo():\n    return 1\n```\n\nThis function returns 1.'
        code, clean = extract_code(text)
        assert clean is True
        assert "def foo():" in code
        assert "This function" not in code

    def test_explanation_before_and_after(self):
        text = (
            "Sure! Here is the code:\n\n"
            "```python\ndef foo():\n    return 1\n```\n\n"
            "The above function returns 1."
        )
        code, clean = extract_code(text)
        assert clean is True
        assert "def foo():" in code
        assert "Sure!" not in code
        assert "above function" not in code

    def test_multiple_code_blocks_picks_correct_one(self):
        text = (
            "```python\nimport os\n```\n\n"
            "```python\ndef my_func():\n    return 42\n```"
        )
        code, clean = extract_code(text, expected_function_name="my_func")
        assert clean is True
        assert "def my_func():" in code

    def test_multiple_code_blocks_default_first(self):
        text = (
            "```python\ndef first():\n    pass\n```\n\n"
            "```python\ndef second():\n    pass\n```"
        )
        code, clean = extract_code(text)
        assert clean is True
        assert "def first():" in code

    def test_empty_output(self):
        code, clean = extract_code("")
        assert code == ""
        assert clean is False

    def test_whitespace_only_output(self):
        code, clean = extract_code("   \n\n  ")
        assert code == ""
        assert clean is False

    def test_none_text_like_empty(self):
        # Bare text with no function def falls back to full text
        code, clean = extract_code("just some explanation text")
        assert clean is False
        assert code == "just some explanation text"

    def test_bare_function_with_leading_text(self):
        text = "Here is the code:\ndef bar(x):\n    return x + 1"
        code, clean = extract_code(text)
        assert clean is True
        assert "def bar(x):" in code
        assert "Here is" not in code


class TestCheckCompliance:
    def test_matching_function_name(self):
        code = "def foo(x):\n    return x"
        assert check_compliance(code, "foo") is True

    def test_non_matching_function_name(self):
        code = "def bar(x):\n    return x"
        assert check_compliance(code, "foo") is False

    def test_empty_code(self):
        assert check_compliance("", "foo") is False

    def test_empty_function_name(self):
        assert check_compliance("def foo(): pass", "") is False

    def test_function_name_in_body_not_def(self):
        code = "def bar():\n    foo = 1\n    return foo"
        assert check_compliance(code, "foo") is False

    def test_function_with_type_hints(self):
        code = "def foo(x: int) -> int:\n    return x"
        assert check_compliance(code, "foo") is True
