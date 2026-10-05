from __future__ import annotations

import argparse

from .engine import LiteraryAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a local question about the indexed novels.")
    parser.add_argument("question", help="Question about one or more novels")
    parser.add_argument("--work", action="append", dest="works", help="work_id to search (repeatable)")
    parser.add_argument("--top-k", type=int, default=4, help="number of passages supplied to the model")
    parser.add_argument(
        "--through-chapter", type=int,
        help="hide passages after this chapter (spoiler protection)",
    )
    parser.add_argument(
        "--mode",
        choices=["Ask", "Explain", "Analyze", "Summarize", "Characters", "Quotes / Search", "Compare"],
        default="Ask",
        help="reading mode",
    )
    args = parser.parse_args()

    assistant = LiteraryAssistant()
    answer = assistant.answer(
        args.question,
        work_ids=args.works,
        top_k=args.top_k,
        max_chapter=args.through_chapter,
        mode=args.mode,
    )
    print(answer.text)
    if answer.citations:
        print("\nSources :")
        for passage in answer.citations:
            print(f"- {passage.citation_label}")


if __name__ == "__main__":
    main()
