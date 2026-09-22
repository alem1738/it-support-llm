
import argparse
import csv
import json
import re
import time

import torch

from load_model import MODEL_NAME, load_model_and_tokenizer, run_inference

CATEGORIES = {
    "Network / Wi-Fi", "VPN", "Account / Authentication", "Microsoft 365",
    "Hardware", "Software", "Printer", "Security", "Mobile Device", "File / Permissions",
}
PRIORITIES = {"Low", "Medium", "High", "Critical"}
REQUIRED_FIELDS = {"category", "priority", "summary", "troubleshooting", "escalation"}

ESCALATION_TARGETS = [
    "Network Operations",
    "Identity & Access Management",
    "Microsoft 365 / Cloud Services",
    "Desktop Support / Hardware",
    "Security / Incident Response",
    "Facilities",
    "No escalation needed",
]


def extract_json(text):
   
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break

    return None


def matched_escalation_targets(text):
    
    if not isinstance(text, str):
        return set()
    text_lower = text.lower()
    matches = set()
    for target in ESCALATION_TARGETS:
        if target.lower() in text_lower:
            matches.add(target)
    if "no escalation" in text_lower:
        matches.add("No escalation needed")
    return matches


def score_example(gold_output, raw_prediction):
    
    predicted = extract_json(raw_prediction)
    json_valid = isinstance(predicted, dict)

    result = {
        "raw_output": raw_prediction,
        "json_valid": json_valid,
        "required_fields_complete": False,
        "category_correct": False,
        "priority_correct": False,
        "escalation_correct": False,
        "predicted_category": None,
        "predicted_priority": None,
        "predicted_escalation": None,
    }

    if not json_valid:
        return result

    present_fields = {
        field for field in REQUIRED_FIELDS
        if field in predicted and predicted[field] not in (None, "", [])
    }
    result["required_fields_complete"] = present_fields == REQUIRED_FIELDS

    result["predicted_category"] = predicted.get("category")
    result["predicted_priority"] = predicted.get("priority")
    result["predicted_escalation"] = predicted.get("escalation")

    result["category_correct"] = predicted.get("category") == gold_output.get("category")
    result["priority_correct"] = predicted.get("priority") == gold_output.get("priority")

    gold_targets = matched_escalation_targets(gold_output.get("escalation", ""))
    predicted_targets = matched_escalation_targets(predicted.get("escalation", ""))
    result["escalation_correct"] = bool(gold_targets) and bool(gold_targets & predicted_targets)

    return result


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(model, tokenizer, records, max_new_tokens=300, system_prompt=None):
    
    from tqdm.auto import tqdm

    rows = []
    bar = tqdm(records, desc="Evaluating", dynamic_ncols=True)
    running_correct = {"category": 0, "priority": 0, "json": 0}
    for i, record in enumerate(bar):
        raw_prediction = run_inference(
            model, tokenizer, record["instruction"], record["input"],
            max_new_tokens=max_new_tokens, system_prompt=system_prompt,
        )
        score = score_example(record["output"], raw_prediction)

        row = {
            "input": record["input"],
            "gold_category": record["output"]["category"],
            "gold_priority": record["output"]["priority"],
            "gold_escalation": record["output"]["escalation"],
            **score,
        }
        rows.append(row)

        running_correct["category"] += int(row["category_correct"])
        running_correct["priority"] += int(row["priority_correct"])
        running_correct["json"] += int(row["json_valid"])
        n_so_far = i + 1
        bar.set_postfix({
            "cat_acc": f"{running_correct['category'] / n_so_far:.1%}",
            "pri_acc": f"{running_correct['priority'] / n_so_far:.1%}",
            "json_ok": f"{running_correct['json'] / n_so_far:.1%}",
            "vram_gb": f"{torch.cuda.memory_allocated() / 1e9:.1f}" if torch.cuda.is_available() else "n/a",
        })
    return rows


def aggregate_metrics(rows):
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n_examples": n,
        "json_validity_rate": sum(r["json_valid"] for r in rows) / n,
        "required_field_completion_rate": sum(r["required_fields_complete"] for r in rows) / n,
        "category_accuracy": sum(r["category_correct"] for r in rows) / n,
        "priority_accuracy": sum(r["priority_correct"] for r in rows) / n,
        "escalation_accuracy": sum(r["escalation_correct"] for r in rows) / n,
    }


def failure_counts(rows):
    n = len(rows)
    return {
        "n_examples": n,
        "invalid_json": sum(not r["json_valid"] for r in rows),
        "missing_required_fields": sum(not r["required_fields_complete"] for r in rows),
        "wrong_category": sum(not r["category_correct"] for r in rows),
        "wrong_priority": sum(not r["priority_correct"] for r in rows),
        "wrong_escalation": sum(not r["escalation_correct"] for r in rows),
    }


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a JSONL file to evaluate against (e.g. data/test.jsonl).")
    parser.add_argument("--output", required=True, help="Path to write per-example results CSV.")
    parser.add_argument("--lora-path", default=None, help="Optional path to a PEFT LoRA adapter to load on top of the base model.")
    parser.add_argument("--system-prompt-file", default=None,
                         help="Optional path to a system prompt file (e.g. prompts/baseline_system_prompt.txt) "
                              "prepended for this condition. Omit for the minimal training-format prompt.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples (for smoke testing).")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.limit:
        records = records[:args.limit]

    system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read()

    print(f"Loading {MODEL_NAME}" + (f" with LoRA adapter {args.lora_path}" if args.lora_path else " (base model)") + "...")
    print("System prompt: " + (args.system_prompt_file if system_prompt else "(none — minimal prompt)"))
    model, tokenizer = load_model_and_tokenizer()

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Evaluating {len(records)} example(s) from {args.data}...")
    start_time = time.time()
    rows = evaluate(model, tokenizer, records, max_new_tokens=args.max_new_tokens, system_prompt=system_prompt)
    elapsed = time.time() - start_time

    write_csv(rows, args.output)
    print(f"\nWrote {len(rows)} result rows to {args.output}")
    print(f"Evaluation runtime: {elapsed:.1f}s ({elapsed / max(len(rows), 1):.2f}s/example)")
    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB allocated / "
              f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB reserved")

    metrics = aggregate_metrics(rows)
    print("\n=== Aggregate metrics ===")
    for key, value in metrics.items():
        if key == "n_examples":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.1%}")

    failures = failure_counts(rows)
    print("\n=== Failure counts ===")
    for key, value in failures.items():
        if key != "n_examples":
            print(f"  {key}: {value}/{failures['n_examples']}")


if __name__ == "__main__":
    main()
