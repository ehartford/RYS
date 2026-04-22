import os
import pickle
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
for name in [
      "coarse_combined_results.pkl",
      "coarse_math_results.pkl",
      "coarse_eq_results.pkl",
]:
      path = root / name
      data = pickle.loads(path.read_bytes())
      print(name, len(data))
