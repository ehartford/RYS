import os
import pickle
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
for name in [
      "dense_37_49_combined_results.pkl",
      "dense_37_49_math_results.pkl",
      "dense_37_49_eq_results.pkl",
]:
      path = root / name
      data = pickle.loads(path.read_bytes())
      print(name, len(data))
