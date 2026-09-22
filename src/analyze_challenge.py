

from collections import Counter
import csv

from analyze_results import (
    METRIC_LABELS, load_results, rate, failure_type, write_error_analysis_csv,
)
from evaluate import matched_escalation_targets

BASELINE_PATH = "evaluation/challenge_baseline_results.csv"
FINE_TUNED_PATH = "evaluation/challenge_fine_tuned_results.csv"
PAIRED_OUTPUT_PATH = "evaluation/challenge_paired_analysis.csv"
ERROR_OUTPUT_PATH = "evaluation/challenge_error_analysis.csv"

METRIC_FIELDS = ["category_correct", "priority_correct", "escalation_correct"]


def print_comparison_table(baseline_rows, fine_tuned_rows):
    print(f"{'Metric':<22} {'Base+Engineered':>16} {'Fine-Tuned+Minimal':>20} {'Abs. Change':>14}")
    print("-" * 74)
    for field, label in METRIC_LABELS:
        b = rate(baseline_rows, field)
        f = rate(fine_tuned_rows, field)
        print(f"{label:<22} {b:>15.1%} {f:>19.1%} {f - b:>+13.1%}")


def verify_alignment(baseline_rows, fine_tuned_rows):
    if len(baseline_rows) != len(fine_tuned_rows):
        raise SystemExit(
            f"Row count mismatch: baseline has {len(baseline_rows)}, fine-tuned has {len(fine_tuned_rows)}. "
            f"Cannot perform paired analysis."
        )
    mismatches = [
        i for i, (b, f) in enumerate(zip(baseline_rows, fine_tuned_rows))
        if b["input"] != f["input"]
    ]
    if mismatches:
        raise SystemExit(
            f"Row alignment mismatch at {len(mismatches)} position(s) (first: index {mismatches[0]}). "
            f"Baseline and fine-tuned CSVs must be evaluated against the exact same ordered example list."
        )


def bucket_for(baseline_correct, fine_tuned_correct):
    if baseline_correct and fine_tuned_correct:
        return "both_correct"
    if fine_tuned_correct and not baseline_correct:
        return "fine_tuned_only"
    if baseline_correct and not fine_tuned_correct:
        return "baseline_only"
    return "both_wrong"


def escalation_target(text):
    matched = matched_escalation_targets(text)
    if not matched:
        return "(none matched)"
    return sorted(matched)[0]


def paired_bucket_counts(baseline_rows, fine_tuned_rows):
    counts = {field: Counter() for field in METRIC_FIELDS}
    for b, f in zip(baseline_rows, fine_tuned_rows):
        for field in METRIC_FIELDS:
            counts[field][bucket_for(b[field], f[field])] += 1
    return counts


def build_confusions(baseline_rows, fine_tuned_rows):
    def cat_confusions(rows):
        c = Counter()
        for r in rows:
            if not r["category_correct"]:
                c[(r["gold_category"], r["predicted_category"])] += 1
        return c

    def pri_confusions(rows):
        c = Counter()
        for r in rows:
            if not r["priority_correct"]:
                c[(r["gold_priority"], r["predicted_priority"])] += 1
        return c

    def esc_confusions(rows):
        c = Counter()
        for r in rows:
            if not r["escalation_correct"]:
                gold_t = escalation_target(r["gold_escalation"])
                pred_t = escalation_target(r["predicted_escalation"])
                c[(gold_t, pred_t)] += 1
        return c

    return {
        "category": {"baseline": cat_confusions(baseline_rows), "fine_tuned": cat_confusions(fine_tuned_rows)},
        "priority": {"baseline": pri_confusions(baseline_rows), "fine_tuned": pri_confusions(fine_tuned_rows)},
        "escalation": {"baseline": esc_confusions(baseline_rows), "fine_tuned": esc_confusions(fine_tuned_rows)},
    }


def print_bucket_report(counts):
    print("\n=== Paired example-level buckets (base+engineered vs fine-tuned+minimal) ===")
    for field in METRIC_FIELDS:
        c = counts[field]
        total = sum(c.values())
        print(f"\n{field}:")
        for bucket in ["both_correct", "fine_tuned_only", "baseline_only", "both_wrong"]:
            n = c.get(bucket, 0)
            print(f"  {bucket:<16}: {n:>3} ({n/total:.1%})")


def print_confusion_report(confusions):
    for metric_name, by_condition in confusions.items():
        print(f"\n=== {metric_name.capitalize()} confusions (gold -> predicted) ===")
        for condition, counter in by_condition.items():
            print(f"\n  [{condition}]")
            if not counter:
                print("    None.")
            for (gold, pred), count in counter.most_common(8):
                print(f"    {gold!r} -> {pred!r}: {count}")


def print_regression_and_improvement_examples(baseline_rows, fine_tuned_rows, field, gold_field, base_pred_field, ft_pred_field, label, limit=10):
    print(f"\n=== {label}: examples where fine-tuning REGRESSED (base correct, fine-tuned wrong) ===")
    n = 0
    for b, f in zip(baseline_rows, fine_tuned_rows):
        if b[field] and not f[field]:
            n += 1
            if n <= limit:
                print(f"  input: {b['input'][:90]!r}")
                print(f"    gold={b[gold_field]!r}  base_pred={b[base_pred_field]!r}  ft_pred={f[ft_pred_field]!r}")
    print(f"  Total regressions: {n}")

    print(f"\n=== {label}: examples where fine-tuning FIXED a base error (base wrong, fine-tuned correct) ===")
    n = 0
    for b, f in zip(baseline_rows, fine_tuned_rows):
        if not b[field] and f[field]:
            n += 1
            if n <= limit:
                print(f"  input: {b['input'][:90]!r}")
                print(f"    gold={b[gold_field]!r}  base_pred={b[base_pred_field]!r}  ft_pred={f[ft_pred_field]!r}")
    print(f"  Total fixes: {n}")


def write_paired_csv(baseline_rows, fine_tuned_rows, path):
    fieldnames = [
        "input", "gold_category", "gold_priority", "gold_escalation",
        "baseline_category", "fine_tuned_category", "category_bucket",
        "baseline_priority", "fine_tuned_priority", "priority_bucket",
        "baseline_escalation_target", "fine_tuned_escalation_target", "escalation_bucket",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for b, f in zip(baseline_rows, fine_tuned_rows):
            writer.writerow({
                "input": b["input"],
                "gold_category": b["gold_category"],
                "gold_priority": b["gold_priority"],
                "gold_escalation": b["gold_escalation"],
                "baseline_category": b["predicted_category"],
                "fine_tuned_category": f["predicted_category"],
                "category_bucket": bucket_for(b["category_correct"], f["category_correct"]),
                "baseline_priority": b["predicted_priority"],
                "fine_tuned_priority": f["predicted_priority"],
                "priority_bucket": bucket_for(b["priority_correct"], f["priority_correct"]),
                "baseline_escalation_target": escalation_target(b["predicted_escalation"]),
                "fine_tuned_escalation_target": escalation_target(f["predicted_escalation"]),
                "escalation_bucket": bucket_for(b["escalation_correct"], f["escalation_correct"]),
            })
    print(f"\nWrote paired per-example analysis to {path}")


def main():
    print("=" * 74)
    print("NOVEL / OUT-OF-TEMPLATE CHALLENGE EVALUATION")
    print("(75-example data/challenge.jsonl — distinct from the 230-example closed-set benchmark)")
    print("=" * 74)

    baseline_rows = load_results(BASELINE_PATH)
    fine_tuned_rows = load_results(FINE_TUNED_PATH)
    verify_alignment(baseline_rows, fine_tuned_rows)

    print(f"\nBaseline: {BASELINE_PATH} ({len(baseline_rows)} examples)")
    print(f"Fine-tuned: {FINE_TUNED_PATH} ({len(fine_tuned_rows)} examples)")
    print()
    print_comparison_table(baseline_rows, fine_tuned_rows)

    counts = paired_bucket_counts(baseline_rows, fine_tuned_rows)
    print_bucket_report(counts)

    confusions = build_confusions(baseline_rows, fine_tuned_rows)
    print_confusion_report(confusions)

    print_regression_and_improvement_examples(
        baseline_rows, fine_tuned_rows, "category_correct", "gold_category",
        "predicted_category", "predicted_category", "CATEGORY")
    print_regression_and_improvement_examples(
        baseline_rows, fine_tuned_rows, "priority_correct", "gold_priority",
        "predicted_priority", "predicted_priority", "PRIORITY")
    print_regression_and_improvement_examples(
        baseline_rows, fine_tuned_rows, "escalation_correct", "gold_escalation",
        "predicted_escalation", "predicted_escalation", "ESCALATION")

    write_paired_csv(baseline_rows, fine_tuned_rows, PAIRED_OUTPUT_PATH)

    failed = [r for r in fine_tuned_rows if failure_type(r)]
    write_error_analysis_csv(failed, ERROR_OUTPUT_PATH)


if __name__ == "__main__":
    main()
