#!/usr/bin/env python3
"""Generate contrastive typed/untyped code dataset.

Uses Mistral API to generate typed code snippets in 10 categories across
TypeScript and Python, then programmatically strips types to create matched
untyped variants. This guarantees the ONLY difference is type annotations.

Requires MISTRAL_API_KEY environment variable.

Usage:
    python 00_generate_dataset.py [--per-category 10] [--model mistral-medium-latest]
"""

import argparse
import ast
import json
import os
import re
import time

from mistralai import Mistral


CATEGORIES = [
    {
        "name": "simple_functions",
        "ts_desc": "simple TypeScript functions with primitive type annotations (number, string, boolean). Include parameter types and return types.",
        "py_desc": "simple Python functions with type annotations for primitives (int, float, str, bool). Include parameter types and return type hints.",
    },
    {
        "name": "data_structures",
        "ts_desc": "TypeScript interfaces and types for data structures (e.g., User, Config, Point). Use interface definitions with typed fields.",
        "py_desc": "Python dataclasses or TypedDict definitions with type annotations for all fields.",
    },
    {
        "name": "generics",
        "ts_desc": "TypeScript generic functions and classes using type parameters like <T>, <K, V>. Include generic constraints.",
        "py_desc": "Python generic functions using TypeVar or the newer T syntax. Include Generic base classes.",
    },
    {
        "name": "union_types",
        "ts_desc": "TypeScript functions using union types (string | number), optional parameters (?:), and null checks.",
        "py_desc": "Python functions using Union types, Optional, and None checks with type annotations.",
    },
    {
        "name": "classes",
        "ts_desc": "TypeScript classes with typed members, access modifiers (public/private), typed constructor parameters, and typed methods.",
        "py_desc": "Python classes with type annotations on all attributes, __init__ parameters, and method signatures.",
    },
    {
        "name": "higher_order_functions",
        "ts_desc": "TypeScript higher-order functions with typed callback parameters like (fn: (x: T) => U). Use arrow function types.",
        "py_desc": "Python higher-order functions with Callable type hints for function parameters and return types.",
    },
    {
        "name": "error_handling",
        "ts_desc": "TypeScript functions with typed error handling, using union return types like Result<T, Error> or string | Error.",
        "py_desc": "Python functions with type annotations on exception handling, using Union return types or custom result types.",
    },
    {
        "name": "algorithms",
        "ts_desc": "TypeScript algorithm implementations using typed containers like Map<string, number[]>, Set<T>, typed arrays.",
        "py_desc": "Python algorithm implementations with typed containers like dict[str, list[int]], set[str], list[tuple[int, int]].",
    },
    {
        "name": "utility_functions",
        "ts_desc": "TypeScript utility/helper functions with full type annotations including complex parameter and return types.",
        "py_desc": "Python utility/helper functions with complete type annotations on all parameters and return values.",
    },
    {
        "name": "api_data_processing",
        "ts_desc": "TypeScript async functions for API/data processing with typed Promises, interfaces for API responses, typed fetch patterns.",
        "py_desc": "Python async functions for API/data processing with typed coroutines, TypedDict for API responses, typed async patterns.",
    },
]


def generate_typed_snippets(
    client: Mistral,
    model: str,
    language: str,
    category: dict,
    count: int,
) -> list[str]:
    """Generate typed code snippets using Mistral API."""
    lang_label = "TypeScript" if language == "typescript" else "Python"
    desc_key = "ts_desc" if language == "typescript" else "py_desc"

    prompt = f"""Generate exactly {count} distinct {lang_label} code snippets for the category: {category[desc_key]}

Requirements:
- Each snippet should be 5-20 lines of code
- Every snippet MUST have full, explicit type annotations on ALL parameters, return types, and variables
- Make snippets diverse within the category — different function names, different logic
- Include ONLY the code, no explanations
- Separate each snippet with a line containing only "---SEPARATOR---"
- Do NOT number the snippets
- Do NOT wrap in markdown code blocks"""

    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Try splitting by ---SEPARATOR--- first (handles clean case)
    snippets = [s.strip() for s in raw.split("---SEPARATOR---") if s.strip()]

    # Fallback: if model wrapped in code blocks, split by ``` boundaries
    if len(snippets) <= 1:
        # Extract content from ```...``` blocks
        code_blocks = re.findall(
            r"```(?:typescript|python|ts|py)?\s*\n(.*?)```",
            raw, re.DOTALL
        )
        if len(code_blocks) > 1:
            snippets = [s.strip() for s in code_blocks if s.strip()]

    # Second fallback: split by blank-line-separated function/class defs
    if len(snippets) <= 1:
        chunks = re.split(r"\n\n(?=(?:def |class |function |async function |interface |type |export |from |import |@dataclass))", raw)
        if len(chunks) > 1:
            snippets = [s.strip() for s in chunks if s.strip()]

    # Clean markdown code fences from any remaining snippets
    cleaned = []
    for s in snippets:
        s = re.sub(r"^```(?:typescript|python|ts|py)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
        # Also clean stray separator fragments
        s = re.sub(r"^---\s*$", "", s, flags=re.MULTILINE)
        s = re.sub(r"^SEPARATOR---\s*$", "", s, flags=re.MULTILINE)
        s = s.strip()
        if s and len(s) > 20:  # Skip fragments too short to be real snippets
            cleaned.append(s)

    return cleaned[:count]


# --- Type stripping: TypeScript -> JavaScript ---

def strip_ts_types(code: str) -> str:
    """Strip TypeScript type annotations to produce valid JavaScript."""
    result = code

    # Remove interface/type declarations (entire block)
    result = re.sub(
        r"^(?:export\s+)?interface\s+\w+(?:<[^>]*>)?\s*\{[^}]*\}\s*;?\s*$",
        "", result, flags=re.MULTILINE | re.DOTALL
    )
    result = re.sub(
        r"^(?:export\s+)?type\s+\w+(?:<[^>]*>)?\s*=\s*[^;]+;\s*$",
        "", result, flags=re.MULTILINE
    )

    # Remove generic type parameters from functions/classes: func<T, U>( -> func(
    result = re.sub(r"(<\s*(?:[A-Z]\w*(?:\s+extends\s+\w+)?(?:\s*,\s*[A-Z]\w*(?:\s+extends\s+\w+)?)*)\s*>)\s*\(", "(", result)

    # Remove generic parameters from class declarations: class Foo<T> -> class Foo
    result = re.sub(r"(class\s+\w+)\s*<[^>]+>", r"\1", result)

    # Remove return type annotations: ): Type { -> ) {    and ): Type => -> ) =>
    result = re.sub(r"\)\s*:\s*[^{=,)]+?(?=\s*\{)", ")", result)
    result = re.sub(r"\)\s*:\s*[^{=>,)]+?(?=\s*=>)", ")", result)

    # Remove parameter type annotations: (name: string, -> (name,
    # Handle complex types like (x: Map<string, number[]>) carefully
    result = re.sub(r"(\w+)\s*\??\s*:\s*(?:[A-Z]\w*(?:<[^>]*>)?(?:\[\])*(?:\s*\|\s*[A-Z]\w*(?:<[^>]*>)?(?:\[\])*)*|string|number|boolean|any|void|null|undefined|never)(?:\[\])*", r"\1", result)

    # Remove 'as Type' assertions
    result = re.sub(r"\s+as\s+\w+(?:<[^>]+>)?", "", result)

    # Remove access modifiers
    result = re.sub(r"\b(public|private|protected|readonly)\s+", "", result)

    # Replace 'const x: Type =' with 'const x ='
    result = re.sub(r"((?:const|let|var)\s+\w+)\s*:\s*[^=]+?(=)", r"\1 \2", result)

    # Clean up empty lines and whitespace
    lines = [line for line in result.split("\n") if line.strip()]
    return "\n".join(lines)


# --- Type stripping: Python typed -> Python untyped ---

class TypeAnnotationStripper(ast.NodeTransformer):
    """AST transformer that removes all type annotations from Python code."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        # Remove return annotation
        node.returns = None
        # Remove parameter annotations
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        node.returns = None
        for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    def visit_AnnAssign(self, node):
        """Convert `x: int = 5` to `x = 5`, or remove `x: int` without value."""
        if node.value is not None:
            return ast.Assign(
                targets=[node.target],
                value=node.value,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        # Bare annotation with no value (e.g., `x: int`) — remove entirely
        return None


def strip_python_types(code: str) -> str:
    """Strip Python type annotations using the AST module."""
    try:
        tree = ast.parse(code)
        stripper = TypeAnnotationStripper()
        tree = stripper.visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except SyntaxError:
        # Fallback: regex-based stripping
        return strip_python_types_regex(code)


def strip_python_types_regex(code: str) -> str:
    """Regex fallback for stripping Python type annotations."""
    result = code
    # Remove return type hints: ) -> Type:
    result = re.sub(r"\)\s*->\s*[^:]+:", "):", result)
    # Remove parameter annotations: param: Type -> param
    result = re.sub(r"(\w+)\s*:\s*(?:int|float|str|bool|list|dict|tuple|set|Optional|Union|Any|None|Callable)\b[^,=)]*", r"\1", result)
    # Remove variable annotations: x: int = -> x =
    result = re.sub(r"^(\s*\w+)\s*:\s*[^=]+?(=)", r"\1 \2", result, flags=re.MULTILINE)
    return result


def strip_types(code: str, language: str) -> str:
    """Strip type annotations based on language."""
    if language == "typescript":
        return strip_ts_types(code)
    else:
        return strip_python_types(code)


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive typed/untyped dataset")
    parser.add_argument("--per-category", type=int, default=10, help="Snippets per category per language")
    parser.add_argument("--model", type=str, default="mistral-medium-latest", help="Mistral model")
    parser.add_argument("--output-dir", type=str, default="scripts/typing/dataset", help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    client = Mistral(api_key=api_key)

    all_records = []
    pairs_manifest = {}
    languages = ["typescript", "python"]

    for language in languages:
        lang_short = "ts" if language == "typescript" else "py"
        lang_family = "ts_js" if language == "typescript" else "python"

        for cat in CATEGORIES:
            cat_name = cat["name"]
            print(f"\n[{language}] Generating {args.per_category} snippets for '{cat_name}'...")

            snippets = generate_typed_snippets(
                client, args.model, language, cat, args.per_category
            )
            print(f"  Got {len(snippets)} snippets")

            for j, typed_code in enumerate(snippets):
                pair_id = f"{lang_short}_{cat_name}_{j:03d}"

                # Strip types to create untyped variant
                untyped_code = strip_types(typed_code, language)

                # Typed record
                typed_record = {
                    "pair_id": pair_id,
                    "category": cat_name,
                    "language": language,
                    "language_family": lang_family,
                    "typing": "typed",
                    "code": typed_code,
                }
                all_records.append(typed_record)

                # Untyped record
                untyped_record = {
                    "pair_id": pair_id,
                    "category": cat_name,
                    "language": language,
                    "language_family": lang_family,
                    "typing": "untyped",
                    "code": untyped_code,
                }
                all_records.append(untyped_record)

                pairs_manifest[pair_id] = {
                    "category": cat_name,
                    "language": language,
                    "language_family": lang_family,
                }

            # Rate limit between category requests
            time.sleep(1)

    # Split into typed/untyped files
    typed_records = [r for r in all_records if r["typing"] == "typed"]
    untyped_records = [r for r in all_records if r["typing"] == "untyped"]

    typed_path = os.path.join(args.output_dir, "typed_snippets.json")
    untyped_path = os.path.join(args.output_dir, "untyped_snippets.json")
    manifest_path = os.path.join(args.output_dir, "pairs_manifest.json")

    with open(typed_path, "w") as f:
        json.dump(typed_records, f, indent=2)
    with open(untyped_path, "w") as f:
        json.dump(untyped_records, f, indent=2)
    with open(manifest_path, "w") as f:
        json.dump(pairs_manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Dataset Generation Complete")
    print(f"{'='*60}")
    print(f"  Typed snippets:   {len(typed_records)} -> {typed_path}")
    print(f"  Untyped snippets: {len(untyped_records)} -> {untyped_path}")
    print(f"  Pairs manifest:   {len(pairs_manifest)} pairs -> {manifest_path}")

    # Print per-category counts
    for language in languages:
        lang_short = "ts" if language == "typescript" else "py"
        print(f"\n  {language}:")
        for cat in CATEGORIES:
            count = sum(
                1 for r in typed_records
                if r["category"] == cat["name"] and r["language"] == language
            )
            print(f"    {cat['name']}: {count} pairs")


if __name__ == "__main__":
    main()
