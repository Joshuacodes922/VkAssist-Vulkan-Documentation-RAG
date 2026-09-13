import argparse
import json
from pathlib import Path

from app.evaluation import evaluate, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid retrieval against labeled relevance judgments.")
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/vulkan_retrieval.json"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=25)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    report = evaluate(load_cases(args.dataset), args.limit, args.candidate_limit, not args.no_rerank)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
