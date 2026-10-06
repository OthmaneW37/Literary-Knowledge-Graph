"""Split by source book before expanding positive/negative training pairs."""
import random


def grouped_split(records, seed=37):
    groups = {}
    for row in records:
        group = row.get('group_id') or row.get('work_id') or row.get('document_id')
        if not isinstance(group, str) or not group.strip():
            raise ValueError('Chaque exemple doit fournir group_id, work_id ou document_id.')
        groups.setdefault(group, []).append(row)
    names = sorted(groups)
    if len(names) < 3:
        raise ValueError('Au moins trois groupes documentaires sont requis pour train/validation/test.')
    random.Random(seed).shuffle(names)
    holdout = max(1, int(len(names) * .15))
    assignments = {'test': names[:holdout], 'validation': names[holdout:2*holdout], 'train': names[2*holdout:]}
    splits = {name: [row for group in ids for row in groups[group]] for name, ids in assignments.items()}
    # Exact duplicates across source groups also leak relevance labels.
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            for passage in [row['positive'], *row['hard_negatives']]:
                key = ' '.join(passage.casefold().split())
                if key in seen and seen[key] != split:
                    raise ValueError('Un passage apparaît dans plusieurs partitions ; corriger les groupes.')
                seen[key] = split
    return splits, assignments
