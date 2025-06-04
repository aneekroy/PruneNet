import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_name: str
    al_method: str
    num_shots: int
    pool_fraction: float
    classification_tasks: list
    multichoice_tasks: list
    raw_data_dir: str
    standardized_data_dir: str
    output_dir: str
    device: str = "cpu"
    seed: int = 42
    max_seq_length: int = 512


def load_cfg(path: str = None) -> Config:
    path = Path(path or Path(__file__).resolve().parent.parent / "config.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

cfg = load_cfg()
