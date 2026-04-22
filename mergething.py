import os
import pickle
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])

pairs = [
    ("coarse_math_results.pkl", "dense_37_49_math_results.pkl", "merged_math_results.pkl"),
    ("coarse_eq_results.pkl", "dense_37_49_eq_results.pkl", "merged_eq_results.pkl"),
    ("coarse_combined_results.pkl", "dense_37_49_combined_results.pkl", "merged_combined_results.pkl"),
]

for a, b, out in pairs:
    merged = {}
    for name in [a, b]:
        path = root / name
        with path.open("rb") as f:
            merged.update(pickle.load(f))
    out_path = root / out
    with out_path.open("wb") as f:
        pickle.dump(merged, f)
    print(out, len(merged))