import pandas as pd
from pathlib import Path
from typing import Tuple, List

from .config import cfg


class CrossFitDataset:
    def __init__(self, task_name: str, split: str):
        assert split in {"pool", "test"}
        path = Path(cfg.standardized_data_dir) / f"{task_name}_{split}.csv"
        self.df = pd.read_csv(path)
        self.task_name = task_name
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if "text" in row:
            return row["text"], row["label"]
        else:
            choices = eval(row["choices"])
            return row["question"], choices, row["label"]

    def get_all_texts(self) -> List[str]:
        if "text" in self.df.columns:
            return self.df["text"].tolist()
        return self.df["question"].tolist()

    def get_all_labels(self) -> List[str]:
        return self.df["label"].astype(str).tolist()
