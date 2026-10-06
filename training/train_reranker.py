from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
try:
    from .splits import grouped_split
    from .build_dataset import validate_records
except ImportError:
    from splits import grouped_split
    from build_dataset import validate_records


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small cross-encoder on approved literary relevance labels.")
    parser.add_argument("dataset", type=Path, help="JSONL from build_dataset.py")
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--output", type=Path, default=Path("models/reranker"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from sentence_transformers import CrossEncoder, InputExample
    except ImportError as exc:
        raise SystemExit("Installez les outils ML avec : python -m pip install -e '.[training]'") from exc

    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < 2:
        raise SystemExit("Fournissez au moins deux exemples réels pour entraîner et valider le modèle.")
    partitions, assignments = grouped_split(validate_records(records))
    def pairs(rows):
        result = []
        for row in rows:
            result.append(InputExample(texts=[row['question'], row['positive']], label=1.0))
            result.extend(InputExample(texts=[row['question'], text], label=0.0) for text in row['hard_negatives'])
        return result
    train_examples, validation_examples = pairs(partitions['train']), pairs(partitions['validation'])
    random.seed(37)
    torch.manual_seed(37)
    model = CrossEncoder(args.base_model, num_labels=1, max_length=512)
    loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.fit(train_dataloader=loader, epochs=args.epochs, optimizer_params={"lr": args.learning_rate},
              loss_fct=torch.nn.BCEWithLogitsLoss(), output_path=str(args.output), show_progress_bar=True)
    scores = model.predict([(item.texts[0], item.texts[1]) for item in validation_examples])
    labels = [item.label for item in validation_examples]
    accuracy = sum((float(score) >= 0.5) == (label >= 0.5) for score, label in zip(scores, labels)) / len(labels)
    report = {"base_model": args.base_model, "trained_model": str(args.output), "examples": len(records),
              "training_pairs": len(train_examples), "validation_pairs": len(validation_examples),
              "validation_accuracy": accuracy, "seed": 37, "split_groups": assignments,
              "held_out_test_examples": len(partitions['test']), "test_used_for_training": False}
    (args.output / 'held_out_test.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in partitions['test']), encoding='utf-8')
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
