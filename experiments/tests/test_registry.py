from pathlib import Path
import pytest
from experiments.src.directional_pairs import load_approved_process_pairs

def test_empty_registry_blocks_real_analysis():
    path = Path("experiments/configs/real_process_pairs.csv")
    with pytest.raises(ValueError, match="No approved"):
        load_approved_process_pairs(path, "swat")
