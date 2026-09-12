"""Validate IT support ticket datasets against dataset_schema.md.

Usage:
    python src/validate_dataset.py --file data/raw/initial_tickets.jsonl
    python src/validate_dataset.py --train data/train.jsonl --validation data/validation.jsonl --test data/test.jsonl

Exits with a non-zero status if any error-level issue is found. Warnings
(e.g. class imbalance) are reported but do not fail the run.
"""

import argparse
import json
import sys
from collections import Counter

CATEGORIES = {
    "Network / Wi-Fi",
    "VPN",
    "Account / Authentication",
    "Microsoft 365",
    "Hardware",
    "Software",
    "Printer",
    "Security",
    "Mobile Device",
    "File / Permissions",
}

PRIORITIES = {"Low", "Medium", "High", "Critical"}

REQUIRED_OUTPUT_KEYS = {"category", "priority", "summary", "troubleshooting", "escalation"}

REQUIRED_TOP_KEYS = {"instruction", "input", "output"}

EXPECTED_INSTRUCTION = "Analyze this IT support ticket."

MAX_INPUT_LEN = 500
MIN_TROUBLESHOOTING_STEPS = 2
IMBALANCE_WARN_RATIO = 0.4  # warn if one class holds more than 40% of examples


def load_jsonl(path):
    """Return (records, errors). Each record is (lineno, parsed_dict)."""
    records = []
    errors = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{lineno}: invalid JSON ({e})")
                continue
            records.append((lineno, obj))
    return records, errors


def validate_record(path, lineno, record):
    """Return a list of error strings for a single record."""
    errors = []
    prefix = f"{path}:{lineno}"

    if not isinstance(record, dict):
        return [f"{prefix}: record is not a JSON object"]

    missing_top = REQUIRED_TOP_KEYS - record.keys()
    if missing_top:
        errors.append(f"{prefix}: missing top-level field(s) {sorted(missing_top)}")
        return errors  # further checks depend on these fields existing

    if record["instruction"] != EXPECTED_INSTRUCTION:
        errors.append(f"{prefix}: unexpected instruction value {record['instruction']!r}")

    ticket_input = record["input"]
    if not isinstance(ticket_input, str) or not ticket_input.strip():
        errors.append(f"{prefix}: empty ticket description")
    elif len(ticket_input) > MAX_INPUT_LEN:
        errors.append(f"{prefix}: input exceeds {MAX_INPUT_LEN} characters ({len(ticket_input)})")

    output = record["output"]
    if not isinstance(output, dict):
        errors.append(f"{prefix}: output is not a JSON object")
        return errors

    missing_fields = REQUIRED_OUTPUT_KEYS - output.keys()
    extra_fields = output.keys() - REQUIRED_OUTPUT_KEYS
    if missing_fields:
        errors.append(f"{prefix}: output missing field(s) {sorted(missing_fields)}")
    if extra_fields:
        errors.append(f"{prefix}: output has unexpected field(s) {sorted(extra_fields)}")

    if "category" in output and output["category"] not in CATEGORIES:
        errors.append(f"{prefix}: invalid category {output['category']!r}")

    if "priority" in output and output["priority"] not in PRIORITIES:
        errors.append(f"{prefix}: invalid priority {output['priority']!r}")

    if "summary" in output and (not isinstance(output["summary"], str) or not output["summary"].strip()):
        errors.append(f"{prefix}: empty or invalid summary")

    if "escalation" in output and (not isinstance(output["escalation"], str) or not output["escalation"].strip()):
        errors.append(f"{prefix}: empty or invalid escalation")

    if "troubleshooting" in output:
        steps = output["troubleshooting"]
        if not isinstance(steps, list) or len(steps) < MIN_TROUBLESHOOTING_STEPS:
            errors.append(
                f"{prefix}: troubleshooting must be a list with at least "
                f"{MIN_TROUBLESHOOTING_STEPS} steps"
            )
        else:
            if any(not isinstance(s, str) or not s.strip() for s in steps):
                errors.append(f"{prefix}: troubleshooting contains an empty step")
            if len(steps) != len(set(steps)):
                errors.append(f"{prefix}: troubleshooting contains duplicate steps")

    return errors


def find_duplicates(path, records):
    """Return errors for duplicate `input` values within a single file."""
    errors = []
    seen = {}
    for lineno, record in records:
        ticket_input = record.get("input")
        if ticket_input is None:
            continue
        if ticket_input in seen:
            errors.append(
                f"{path}:{lineno}: duplicate input (first seen at line {seen[ticket_input]})"
            )
        else:
            seen[ticket_input] = lineno
    return errors


def find_overlap(path_a, records_a, path_b, records_b):
    """Return errors for `input` values shared between two files."""
    inputs_a = {r.get("input") for _, r in records_a if r.get("input") is not None}
    errors = []
    for lineno, record in records_b:
        ticket_input = record.get("input")
        if ticket_input in inputs_a:
            errors.append(
                f"{path_b}:{lineno}: input overlaps with {path_a} ({ticket_input!r})"
            )
    return errors


def check_class_balance(path, records):
    """Return warnings if any category or priority dominates the dataset."""
    warnings = []
    total = len(records)
    if total == 0:
        return warnings

    for field, allowed in (("category", CATEGORIES), ("priority", PRIORITIES)):
        counts = Counter(
            r.get("output", {}).get(field) for _, r in records if isinstance(r.get("output"), dict)
        )
        for value, count in counts.items():
            if value in allowed and count / total > IMBALANCE_WARN_RATIO:
                warnings.append(
                    f"{path}: {field} {value!r} makes up {count}/{total} "
                    f"({count / total:.0%}) of examples — exceeds {IMBALANCE_WARN_RATIO:.0%} threshold"
                )
        missing = allowed - counts.keys()
        if missing:
            warnings.append(f"{path}: no examples for {field} value(s) {sorted(missing)}")

    return warnings


def validate_file(path):
    """Validate a single JSONL file. Returns (records, errors, warnings)."""
    records, errors = load_jsonl(path)
    for lineno, record in records:
        errors.extend(validate_record(path, lineno, record))
    errors.extend(find_duplicates(path, records))
    warnings = check_class_balance(path, records)
    return records, errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Validate a single JSONL file (e.g. the initial dataset).")
    parser.add_argument("--train", help="Path to train.jsonl")
    parser.add_argument("--validation", help="Path to validation.jsonl")
    parser.add_argument("--test", help="Path to test.jsonl")
    args = parser.parse_args()

    if not any([args.file, args.train, args.validation, args.test]):
        parser.error("Provide --file, or one or more of --train/--validation/--test")

    all_errors = []
    all_warnings = []
    split_records = {}

    paths = [("file", args.file), ("train", args.train), ("validation", args.validation), ("test", args.test)]
    for name, path in paths:
        if not path:
            continue
        records, errors, warnings = validate_file(path)
        split_records[name] = (path, records)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if "train" in split_records and "test" in split_records:
        train_path, train_records = split_records["train"]
        test_path, test_records = split_records["test"]
        all_errors.extend(find_overlap(train_path, train_records, test_path, test_records))

    if "validation" in split_records and "test" in split_records:
        val_path, val_records = split_records["validation"]
        test_path, test_records = split_records["test"]
        all_errors.extend(find_overlap(val_path, val_records, test_path, test_records))

    if "train" in split_records and "validation" in split_records:
        train_path, train_records = split_records["train"]
        val_path, val_records = split_records["validation"]
        all_errors.extend(find_overlap(train_path, train_records, val_path, val_records))

    total_records = sum(len(records) for _, records in split_records.values())
    print(f"Validated {total_records} record(s) across {len(split_records)} file(s).\n")

    if all_warnings:
        print(f"WARNINGS ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  - {w}")
        print()

    if all_errors:
        print(f"ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"  - {e}")
        print()
        print("Validation FAILED.")
        sys.exit(1)

    print("Validation PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
