# Active In-Context Learning with Single-Step Active Learning

This document outlines the design of the repository. The goal is to study how different active-learning strategies can be used to select few-shot examples for large language models.

The repository is structured into three main parts:

1. **Data Preprocessing** – scripts to download and standardize the CrossFit tasks.
2. **Core Source Code** – loaders, prompt builders, model wrappers, and active-learning samplers.
3. **Experiments** – small scripts that iterate over tasks to reproduce our results.

Each sampler implements a different heuristic for picking the most useful in-context examples. The main experiment script ties everything together and reports accuracy and (for binary classification) F1 score.
