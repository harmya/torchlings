"""Parse and format pytest output into friendly messages."""

import re
import click


def format_test_output(stdout: str, stderr: str) -> tuple[bool, str]:
    """Parse pytest output and return (passed, friendly_message)."""
    lines = stdout.splitlines()

    passed_tests = []
    failed_tests = []
    seen = set()

    for line in lines:
        # Only parse the verbose test result lines (with PASSED/FAILED markers)
        # These look like: "path/file.py::test_name PASSED  [ 50%]"
        match = re.match(r".*::(\w+)\s+(PASSED|FAILED)", line)
        if match:
            name = match.group(1)
            status = match.group(2)
            if name not in seen:
                seen.add(name)
                if status == "PASSED":
                    passed_tests.append(name)
                else:
                    failed_tests.append(name)

    all_passed = len(failed_tests) == 0 and len(passed_tests) > 0

    if all_passed:
        parts = [click.style("All tests passed!", fg="green", bold=True)]
        for t in passed_tests:
            fn_name = _test_to_fn_name(t)
            parts.append("  " + click.style(fn_name, fg="green"))
        return True, "\n".join(parts)

    # Build friendly failure message
    output_parts = []

    for t in passed_tests:
        fn_name = _test_to_fn_name(t)
        output_parts.append("  " + click.style(fn_name, fg="green"))

    for test_name in failed_tests:
        fn_name = _test_to_fn_name(test_name)
        error_detail = _extract_error_for_test(test_name, stdout)
        friendly = _make_friendly(fn_name, error_detail)
        output_parts.append(
            "  " + click.style(fn_name, fg="red") + " -- " + friendly
        )

    return False, "\n".join(output_parts)


def _test_to_fn_name(test_name: str) -> str:
    """Convert test_foo_bar to foo_bar."""
    if test_name.startswith("test_"):
        return test_name[5:]
    return test_name


def _extract_error_for_test(test_name: str, stdout: str) -> str:
    """Pull the E-lines (assertion details) for a specific test failure."""
    lines = stdout.splitlines()
    in_section = False
    e_lines = []

    for line in lines:
        # Match the test section header: ____ test_name ____
        if test_name in line and "____" in line:
            in_section = True
            continue
        if in_section:
            stripped = line.strip()
            if stripped.startswith("E "):
                e_lines.append(stripped[2:].strip())
            elif stripped.startswith("____") or stripped.startswith("===="):
                break

    return "\n".join(e_lines)


def _make_friendly(fn_name: str, error_detail: str) -> str:
    """Turn raw assertion errors into friendly messages."""
    if not error_detail:
        return "failed"

    # None return - most common for unsolved exercises
    if "NoneType" in error_detail or "is not None" in error_detail or "assert None" in error_detail:
        return "returns None -- fill in the TODO"

    # Class not implemented (just has pass)
    if "TypeError" in error_detail and ("takes no arguments" in error_detail or "object is not callable" in error_detail):
        return "class not implemented yet -- add __init__ and forward"

    if "has no attribute" in error_detail:
        match = re.search(r"has no attribute '(\w+)'", error_detail)
        if match:
            return f"missing attribute '{match.group(1)}'"
        return "missing attribute"

    # Shape mismatch
    shape_match = re.search(r"torch\.Size\((\[.*?\])\).*?torch\.Size\((\[.*?\])\)", error_detail)
    if shape_match:
        return f"wrong shape: got {shape_match.group(1)}, expected {shape_match.group(2)}"

    # Wrong values
    if "assert" in error_detail.lower():
        return _clean_assertion(error_detail)

    # Fallback
    first_line = error_detail.splitlines()[0] if error_detail else "failed"
    return first_line


def _clean_assertion(error_detail: str) -> str:
    """Clean up an assertion error into readable form."""
    lines = error_detail.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("assert "):
            return line[7:]
        if "AssertionError:" in line:
            return line.split("AssertionError:", 1)[1].strip()
        if "AssertionError" not in line and line:
            return line
    return lines[0] if lines else "assertion failed"
