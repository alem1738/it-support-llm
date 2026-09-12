"""Compare baseline vs fine-tuned evaluation results and produce error analysis.

Reads two evaluation CSVs produced by evaluate.py (same test set, different
conditions), computes aggregate metrics for each programmatically, prints a
comparison table with absolute change, and writes a breakdown of every
fine-tuned failure to evaluation/error_analysis.csv.

Usage:
    python src/analyze_results.py \\
        --baseline evaluation/baseline_results.csv \\
        --fine-tuned evaluation/fine_tuned_results.csv \\
        --error-analysis-output evaluation/error_analysis.csv
"""

import argparse
import csv
from collections import Counter

BOOL_FIELDS = ["json_valid", "required_fields_complete", "category_correct",
               "priority_correct", "escalation_correct"]

METRIC_LABELS = [
    ("json_valid", "JSON validity"),
    ("required_fields_complete", "Required fields"),
    ("category_correct", "Category accuracy"),
    ("priority_correct", "Priority accuracy"),
    ("escalation_correct", "Escalation accuracy"),
]


def load_results(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in BOOL_FIELDS:
                row[field] = row[field] == "True"
            rows.append(row)
    return rows


def rate(rows, field):
    n = len(rows)
    return sum(r[field] for r in rows) / n if n else 0.0


def print_comparison_table(baseline_rows, fine_tuned_rows):
    print(f"{'Metric':<22} {'Base+Engineered':>16} {'Fine-Tuned+Minimal':>20} {'Abs. Change':>14}")
    print("-" * 74)
    for field, label in METRIC_LABELS:
        b = rate(baseline_rows, field)
        f = rate(fine_tuned_rows, field)
        change = f - b
        print(f"{label:<22} {b:>15.1%} {f:>19.1%} {change:>+13.1%}")


def failure_type(row):
    types = []
    if not row["json_valid"]:
        types.append("invalid_json")
    if not row["required_fields_complete"]:
        types.append("missing_required_field")
    if not row["category_correct"]:
        types.append("wrong_category")
    if not row["priority_correct"]:
        types.append("wrong_priority")
    if not row["escalation_correct"]:
        types.append("wrong_escalation")
    return types


def error_analysis(fine_tuned_rows):
    failed = [r for r in fine_tuned_rows if failure_type(r)]

    type_counts = Counter()
    for r in failed:
        for t in failure_type(r):
            type_counts[t] += 1

    category_confusions = Counter()
    priority_confusions = Counter()
    for r in fine_tuned_rows:
        if not r["category_correct"]:
            category_confusions[(r["gold_category"], r["predicted_category"])] += 1
        if not r["priority_correct"]:
            priority_confusions[(r["gold_priority"], r["predicted_priority"])] += 1

    return failed, type_counts, category_confusions, priority_confusions


def write_error_analysis_csv(failed_rows, path):
    if not failed_rows:
        print(f"No failures to write to {path}.")
        return
    fieldnames = ["input", "gold_category", "predicted_category", "gold_priority",
                  "predicted_priority", "gold_escalation", "predicted_escalation",
                  "failure_type", "raw_output"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in failed_rows:
            writer.writerow({
                "input": r["input"],
                "gold_category": r["gold_category"],
                "predicted_category": r["predicted_category"],
                "gold_priority": r["gold_priority"],
                "predicted_priority": r["predicted_priority"],
                "gold_escalation": r["gold_escalation"],
                "predicted_escalation": r["predicted_escalation"],
                "failure_type": ";".join(failure_type(r)),
                "raw_output": r["raw_output"],
            })
    print(f"Wrote {len(failed_rows)} failure rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--fine-tuned", required=True)
    parser.add_argument("--error-analysis-output", default=None,
                         help="Optional path to write failure breakdown CSV for the fine-tuned results.")
    args = parser.parse_args()

    baseline_rows = load_results(args.baseline)
    fine_tuned_rows = load_results(args.fine_tuned)

    if len(baseline_rows) != len(fine_tuned_rows):
        print(f"WARNING: baseline has {len(baseline_rows)} rows, fine-tuned has {len(fine_tuned_rows)} rows. "
              f"Comparison assumes the same test set was used for both.")

    print(f"Baseline: {args.baseline} ({len(baseline_rows)} examples)")
    print(f"Fine-tuned: {args.fine_tuned} ({len(fine_tuned_rows)} examples)")
    print()
    print_comparison_table(baseline_rows, fine_tuned_rows)

    failed, type_counts, category_confusions, priority_confusions = error_analysis(fine_tuned_rows)

    print(f"\n=== Fine-tuned error analysis: {len(failed)}/{len(fine_tuned_rows)} example(s) had at least one failure ===")
    for failure_name, count in type_counts.most_common():
        print(f"  {failure_name}: {count}")

    if category_confusions:
        print("\nCategory confusions (gold -> predicted):")
        for (gold, pred), count in category_confusions.most_common():
            print(f"  {gold!r} -> {pred!r}: {count}")
    else:
        print("\nNo category confusions.")

    if priority_confusions:
        print("\nPriority confusions (gold -> predicted):")
        for (gold, pred), count in priority_confusions.most_common():
            print(f"  {gold!r} -> {pred!r}: {count}")
    else:
        print("\nNo priority confusions.")

    if args.error_analysis_output:
        write_error_analysis_csv(failed, args.error_analysis_output)


if __name__ == "__main__":
    main()
