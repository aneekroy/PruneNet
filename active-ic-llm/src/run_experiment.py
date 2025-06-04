import argparse
from typing import List
from pathlib import Path

from .config import cfg
from .data_loader import CrossFitDataset
from .prompt_builder import build_classification_prompt, build_multichoice_prompt
from .models.model_utils import ModelUtils
from .al import RandomSampler, DiversitySampler, UncertaintySampler, SimilaritySampler


SAMPLERS = {
    "random": RandomSampler,
    "diversity": DiversitySampler,
    "uncertainty": UncertaintySampler,
    "similarity": SimilaritySampler,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--al_method")
    p.add_argument("--model_name")
    p.add_argument("--num_shots", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    task = args.task
    al_method = args.al_method or cfg.al_method
    model_name = args.model_name or cfg.model_name
    num_shots = args.num_shots or cfg.num_shots

    pool_dataset = CrossFitDataset(task, "pool")
    test_dataset = CrossFitDataset(task, "test")

    mu = ModelUtils(model_name, device=cfg.device)
    sampler = SAMPLERS[al_method]() if al_method != "random" else SAMPLERS[al_method](cfg.seed)

    if al_method in {"random", "diversity", "uncertainty"}:
        demo_indices = sampler.select(pool_dataset, num_shots)
        demos = [pool_dataset[i] for i in demo_indices]

    results = []
    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        if isinstance(item, tuple) and len(item) == 2:
            text, gold = item
            if al_method == "similarity":
                sim_idx = sampler.select_for_one_test(pool_dataset, text, num_shots)
                demos = [pool_dataset[i] for i in sim_idx]
            prompt = build_classification_prompt(demos, text)
            labels = sorted(set(pool_dataset.get_all_labels()))
            pred = mu.predict_classification(prompt, labels)
        else:
            question, choices, gold = item
            if al_method == "similarity":
                sim_idx = sampler.select_for_one_test(pool_dataset, question, num_shots)
                demos = [pool_dataset[i] for i in sim_idx]
            prompt = build_multichoice_prompt(demos, question, choices)
            pred = mu.predict_multichoice(prompt, choices)
        results.append({"id": idx, "prediction": pred, "gold": gold, "correct": pred == gold})

    acc = sum(r["correct"] for r in results) / len(results)
    output = {
        "task": task,
        "model": model_name,
        "al_method": al_method,
        "num_shots": num_shots,
        "accuracy": acc,
        "per_example": results,
    }
    out_dir = Path(cfg.output_dir) / ("multichoice" if len(results[0]) == 3 else "classification") / task / model_name / al_method
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        import json
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
