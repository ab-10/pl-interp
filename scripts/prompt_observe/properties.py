"""Coding property definitions for prompt-and-observe feature discovery.

Each property defines:
  - description: What the property represents
  - target_prompts: Prompts that should elicit the property in generated code
  - control_prompts: Equivalent prompts that should NOT elicit the property
  - detection_patterns: Regex patterns to measure property density in outputs
"""

import re

PROPERTIES = {
    "error_handling": {
        "description": "Exception handling with try/catch/except blocks and error propagation",
        "target_prompts": [
            "Write a Python function that reads a file and handles all possible errors gracefully with try/except blocks",
            "Write a TypeScript function that fetches data from an API with comprehensive error handling",
            "Write a Python function that parses JSON input with detailed error handling for malformed data",
            "Write a function that connects to a database with retry logic and error handling",
            "Write a robust file upload handler that catches and reports every possible failure mode",
            "Implement a safe division function that handles ZeroDivisionError, TypeError, and returns appropriate error messages",
            "Write a Python function that validates user input with multiple try/except blocks for different error types",
            "Write a TypeScript async function with try/catch that handles network timeouts, 404s, and parse errors",
            "Implement a configuration loader that gracefully handles missing files, invalid YAML, and permission errors",
            "Write a function that processes a CSV file with error handling for encoding issues, missing columns, and type mismatches",
        ],
        "control_prompts": [
            "Write a Python function that reads a file and returns its contents",
            "Write a TypeScript function that fetches data from an API",
            "Write a Python function that parses JSON input",
            "Write a function that connects to a database",
            "Write a file upload handler",
            "Implement a division function that returns the quotient of two numbers",
            "Write a Python function that validates user input",
            "Write a TypeScript async function that makes a network request",
            "Implement a configuration loader that reads a YAML file",
            "Write a function that processes a CSV file and returns its rows",
        ],
        "detection_patterns": [
            r"\btry\b",
            r"\bexcept\b",
            r"\bcatch\b",
            r"\bfinally\b",
            r"\braise\b",
            r"\bthrow\b",
            r"\bError\b",
            r"\bException\b",
            r"\berror\b",
        ],
    },
    "type_annotations": {
        "description": "Static type hints and annotations in function signatures and variables",
        "target_prompts": [
            "Write a Python function with full type annotations on all parameters and return type that sorts a list of dictionaries by a key",
            "Write a TypeScript function with explicit types for a binary search on a sorted array",
            "Write a fully typed Python function using Optional, Union, and List that processes user records",
            "Write a TypeScript function with generic type parameters <T> that implements a stack",
            "Write a Python function with type hints including TypeVar and Generic that creates a typed cache",
            "Write a TypeScript function with Record, Partial, and Pick types that transforms API responses",
            "Write a Python function with complete type annotations that merges two dictionaries recursively",
            "Write a TypeScript interface and a function that validates objects against it",
            "Write a Python dataclass with typed fields and a method with full return type annotations",
            "Write a TypeScript function with union types and type guards that parses mixed input",
        ],
        "control_prompts": [
            "Write a Python function that sorts a list of dictionaries by a key",
            "Write a JavaScript function for binary search on a sorted array",
            "Write a Python function that processes user records",
            "Write a JavaScript function that implements a stack",
            "Write a Python function that creates a cache",
            "Write a JavaScript function that transforms API responses",
            "Write a Python function that merges two dictionaries recursively",
            "Write a JavaScript function that validates objects",
            "Write a Python class with fields and a method",
            "Write a JavaScript function that parses mixed input",
        ],
        "detection_patterns": [
            r":\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
            r"->\s*(?:int|float|str|bool|list|dict|tuple|set|None|Any|Optional|Union|Callable)\b",
            r":\s*(?:List|Dict|Tuple|Set)\[",
            r":\s*(?:number|string|boolean|void|never|any|unknown)\b",
            r":\s*(?:Array|Map|Set|Record|Promise|Partial)\s*<",
            r"<[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*>",
            r"\binterface\s+\w+",
            r"\btype\s+\w+\s*=",
        ],
    },
    "functional_style": {
        "description": "Functional programming: map/filter/reduce, pure functions, immutability",
        "target_prompts": [
            "Write Python code using map, filter, and reduce to process a list of numbers",
            "Write a data pipeline using only pure functions and function composition, no mutation",
            "Write TypeScript code using Array.map, filter, and reduce to transform an array of objects",
            "Write a Python function using functools.reduce and lambda to aggregate data without any for loops",
            "Implement a data transformation pipeline using only higher-order functions, no loops or mutation",
            "Write Python code that uses list comprehensions and generator expressions instead of for loops with append",
            "Write a TypeScript solution using method chaining with map/filter/reduce, no temporary variables",
            "Implement a functional approach to tree traversal using recursion and immutable data structures",
            "Write a Python program using itertools functions (chain, groupby, starmap) instead of nested loops",
            "Write a data validation pipeline using function composition where each validator is a pure function",
        ],
        "control_prompts": [
            "Write Python code using for loops to process a list of numbers",
            "Write a data pipeline using classes with mutable state",
            "Write TypeScript code using for loops to transform an array of objects",
            "Write a Python function using a for loop to aggregate data",
            "Implement a data transformation using for loops and temporary variables",
            "Write Python code using for loops with append to build lists",
            "Write a TypeScript solution using for loops and temporary variables",
            "Implement an iterative approach to tree traversal using a stack",
            "Write a Python program using nested for loops",
            "Write a data validation function using if/else chains",
        ],
        "detection_patterns": [
            r"\bmap\s*\(",
            r"\bfilter\s*\(",
            r"\breduce\s*\(",
            r"\blambda\b",
            r"\bitertools\b",
            r"\bfunctools\b",
            r"\.\bmap\b",
            r"\.\bfilter\b",
            r"\.\breduce\b",
            r"\[.+\bfor\b.+\bin\b.+\]",  # list comprehension
        ],
    },
    "recursion": {
        "description": "Recursive algorithms with self-referential function calls",
        "target_prompts": [
            "Write a recursive Python function to compute the nth Fibonacci number",
            "Implement a recursive binary tree traversal (inorder, preorder, postorder)",
            "Write a recursive function to solve the Tower of Hanoi problem",
            "Implement recursive merge sort in Python",
            "Write a recursive function to generate all permutations of a string",
            "Implement a recursive descent parser for arithmetic expressions",
            "Write a recursive function to flatten a deeply nested list",
            "Implement recursive depth-first search on a graph",
            "Write a recursive function to compute the power set of a set",
            "Implement a recursive solution to the N-Queens problem",
        ],
        "control_prompts": [
            "Write an iterative Python function to compute the nth Fibonacci number using a loop",
            "Implement an iterative binary tree traversal using a stack",
            "Write an iterative solution to move disks between pegs using a loop",
            "Implement iterative merge sort in Python using a bottom-up approach",
            "Write an iterative function to generate all permutations of a string",
            "Implement a parser for arithmetic expressions using a loop and stack",
            "Write an iterative function to flatten a deeply nested list using a stack",
            "Implement iterative breadth-first search on a graph using a queue",
            "Write an iterative function to compute the power set of a set",
            "Implement an iterative backtracking solution to the N-Queens problem",
        ],
        "detection_patterns": [
            r"\bdef\s+(\w+)\b.*\n(?:.*\n)*?.*\b\1\s*\(",  # function calling itself
            r"\breturn\s+\w+\s*\(",  # return with function call (common in recursion)
            r"\brecurs",  # recursive/recursion in comments
            r"\bbase\s+case\b",
        ],
    },
    "oop_patterns": {
        "description": "Object-oriented design: classes, inheritance, encapsulation, design patterns",
        "target_prompts": [
            "Write a Python class hierarchy with a base Animal class and Dog, Cat subclasses using inheritance and polymorphism",
            "Implement the Observer design pattern in Python with a Subject base class and concrete observers",
            "Write a TypeScript class with private fields, getters/setters, and a method that uses this",
            "Implement a Strategy pattern in Python with an abstract base class and three concrete strategies",
            "Write a Python class that uses composition with multiple contained objects and delegation",
            "Implement a Factory pattern in Python with a base Product class and concrete product subclasses",
            "Write a TypeScript class hierarchy for shapes (Circle, Rectangle, Triangle) with abstract methods",
            "Implement a Decorator pattern in Python with a base component and multiple decorator classes",
            "Write a Python class with __init__, __repr__, __eq__, and custom methods that manages a collection",
            "Implement a state machine using Python classes where each state is a class with a handle method",
        ],
        "control_prompts": [
            "Write standalone Python functions to describe animals and output their sounds",
            "Write Python functions with callbacks to notify when data changes",
            "Write a TypeScript function that processes and returns an object with computed fields",
            "Write three Python functions that each implement a different sorting strategy",
            "Write Python functions that operate on nested dictionaries",
            "Write a Python function that creates and returns different product dictionaries based on a type argument",
            "Write Python functions to compute area and perimeter for circles, rectangles, and triangles",
            "Write Python functions that wrap other functions to add logging and timing",
            "Write a Python function that manages a list with add, remove, and search operations",
            "Write a Python function with a dictionary to track state transitions",
        ],
        "detection_patterns": [
            r"\bclass\s+\w+",
            r"\bdef\s+__init__\b",
            r"\bself\.\w+",
            r"\bsuper\(\)",
            r"\b(?:ABC|abstractmethod)\b",
            r"\bextends\b",
            r"\bimplements\b",
            r"\b(?:private|protected|public)\b",
            r"\binherit",
        ],
    },
    "async_concurrent": {
        "description": "Asynchronous programming with async/await, promises, concurrent execution",
        "target_prompts": [
            "Write a Python async function using asyncio that fetches multiple URLs concurrently with aiohttp",
            "Write a TypeScript async function that processes an array of items concurrently with Promise.all",
            "Implement an async Python generator that yields results from a paginated API",
            "Write a Python async function that runs three tasks concurrently and returns when the first completes using asyncio.wait",
            "Write a TypeScript function that implements retry logic with exponential backoff using async/await",
            "Implement an async Python function that reads multiple files concurrently using asyncio.gather",
            "Write a TypeScript function that chains multiple async operations with proper await sequencing",
            "Write a Python async context manager that manages a connection pool",
            "Implement an async Python producer-consumer pattern using asyncio.Queue",
            "Write a TypeScript function that implements a semaphore to limit concurrent API requests",
        ],
        "control_prompts": [
            "Write a Python function that fetches multiple URLs one at a time using requests",
            "Write a TypeScript function that processes an array of items one by one in a for loop",
            "Write a Python generator that yields results from a paginated API using requests",
            "Write a Python function that runs three tasks sequentially and returns the first result",
            "Write a TypeScript function that implements retry logic with a for loop and sleep",
            "Write a Python function that reads multiple files sequentially",
            "Write a TypeScript function that chains multiple operations sequentially",
            "Write a Python context manager that manages a single connection",
            "Write a Python function implementing a producer-consumer pattern using a list",
            "Write a TypeScript function that makes API requests one at a time",
        ],
        "detection_patterns": [
            r"\basync\b",
            r"\bawait\b",
            r"\basyncio\b",
            r"\bPromise\b",
            r"\b\.then\s*\(",
            r"\bgather\b",
            r"\bTask\b",
            r"\bconcurrent",
        ],
    },
    "verbose_documentation": {
        "description": "Extensive documentation: docstrings, inline comments, usage examples",
        "target_prompts": [
            "Write a well-documented Python function with a detailed docstring explaining parameters, return values, examples, and edge cases",
            "Write a heavily commented sorting algorithm where every step is explained",
            "Write a Python class with comprehensive docstrings on every method, including type information and usage examples",
            "Write a function with inline comments explaining the algorithm's time and space complexity at each step",
            "Write a thoroughly documented API endpoint with docstring covering request format, response format, errors, and examples",
            "Write a Python module with a module-level docstring, class docstrings, and method docstrings following Google style",
            "Write a binary search with a comment on every line explaining what it does and why",
            "Write a data processing function with verbose logging and comments explaining each transformation",
            "Write a configuration parser with extensive inline documentation of every option and its default value",
            "Write a well-documented linked list implementation with docstrings explaining invariants and complexity",
        ],
        "control_prompts": [
            "Write a Python function concisely with no comments or docstrings",
            "Write a sorting algorithm with no comments",
            "Write a Python class with no docstrings",
            "Write a function with no comments",
            "Write an API endpoint with no documentation",
            "Write a Python module with no docstrings or comments",
            "Write a binary search implementation",
            "Write a data processing function",
            "Write a configuration parser",
            "Write a linked list implementation",
        ],
        "detection_patterns": [
            r'"""',
            r"'''",
            r"#\s+\S",  # inline comments
            r"//\s+\S",  # JS/TS comments
            r"/\*",  # block comments
            r"\bArgs:\b",
            r"\bReturns:\b",
            r"\bRaises:\b",
            r"\bExample",
            r"\bParam",
            r"\b@param\b",
            r"\b@returns?\b",
        ],
    },
    "defensive_coding": {
        "description": "Input validation, assertions, guard clauses, and boundary checks",
        "target_prompts": [
            "Write a Python function with thorough input validation using isinstance checks and ValueError raises for a user registration form",
            "Write a function that validates all inputs at the top with guard clauses before doing any work",
            "Write a Python function with assert statements checking preconditions, postconditions, and invariants",
            "Write a function that sanitizes and validates a URL string, checking scheme, domain, and path components",
            "Write a Python function that processes a list with checks for None, empty list, wrong types, and out-of-range values",
            "Write a function with boundary checks that prevents buffer overflows, integer overflows, and index errors",
            "Write a Python function that validates a configuration dictionary, checking required keys, value types, and valid ranges",
            "Write a function that validates email, phone, and address fields with specific format checks for each",
            "Write a Python function that uses guard clauses to return early for invalid inputs before the main logic",
            "Write a function that validates API request parameters with detailed error messages for each invalid field",
        ],
        "control_prompts": [
            "Write a Python function for a user registration form that processes the inputs directly",
            "Write a function that processes inputs and returns a result",
            "Write a Python function that computes a value",
            "Write a function that processes a URL string",
            "Write a Python function that processes a list and returns a result",
            "Write a function that processes numeric data",
            "Write a Python function that reads values from a dictionary",
            "Write a function that formats email, phone, and address into a string",
            "Write a Python function with the main logic only, no input checking",
            "Write a function that processes API request parameters",
        ],
        "detection_patterns": [
            r"\bassert\b",
            r"\bisinstance\s*\(",
            r"\bValueError\b",
            r"\bTypeError\b",
            r"\bif\s+not\b",
            r"\bif\s+\w+\s+is\s+None\b",
            r"\braise\b",
            r"\bvalidat",
            r"\bcheck\b",
            r"\bguard\b",
        ],
    },
}


def get_detection_regex(property_name: str) -> re.Pattern:
    """Compile detection patterns for a property into a single regex."""
    patterns = PROPERTIES[property_name]["detection_patterns"]
    combined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(combined, re.MULTILINE | re.IGNORECASE)


def compute_property_density(text: str, property_name: str) -> dict:
    """Count property markers and compute density (markers per line)."""
    regex = get_detection_regex(property_name)
    lines = text.strip().split("\n")
    total_lines = max(len(lines), 1)

    matches = regex.findall(text)
    total_markers = len(matches)

    # Per-pattern breakdown
    pattern_counts = {}
    for pattern in PROPERTIES[property_name]["detection_patterns"]:
        count = len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
        if count > 0:
            pattern_counts[pattern[:40]] = count

    return {
        "density": round(total_markers / total_lines, 4),
        "total_markers": total_markers,
        "total_lines": total_lines,
        "pattern_counts": pattern_counts,
    }
