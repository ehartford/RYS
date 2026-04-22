from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.beam_search import expand_multi_block_config, load_pair_score_map


class BeamSearchSeedLoaderTests(unittest.TestCase):
    def test_load_pair_score_map_accepts_canonical_vllm_keys(self):
        num_layers = 7
        baseline = tuple(range(num_layers))
        block_2_5 = tuple(expand_multi_block_config(num_layers, ((2, 5),)))
        block_1_3 = tuple(expand_multi_block_config(num_layers, ((1, 3),)))
        multi_block = tuple(expand_multi_block_config(num_layers, ((1, 3), (4, 6))))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.pkl"
            with path.open("wb") as f:
                pickle.dump(
                    {
                        baseline: {"score": 0.5},
                        block_2_5: {"score": 0.7},
                        "3,6": 0.8,
                        "layers:" + ",".join(str(x) for x in block_1_3): {"math_score": 0.9},
                        multi_block: {"score": 1.0},
                    },
                    f,
                )

            loaded = load_pair_score_map(path, num_layers=num_layers)

        self.assertEqual(loaded[(0, 0)], 0.5)
        self.assertEqual(loaded[(2, 5)], 0.7)
        self.assertEqual(loaded[(3, 6)], 0.8)
        self.assertEqual(loaded[(1, 3)], 0.9)
        self.assertNotIn(1.0, loaded.values())


if __name__ == "__main__":
    unittest.main()
