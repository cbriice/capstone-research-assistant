import itertools
import copy
import csv

from pipeline.config import load_config
from pipeline.train import run_training

def main():
    cfg = load_config()

    #start with a small search space
    search_space = {
        "d_model": [8, 16, 32],
        "nhead": [1, 2, 4],
        "num_layers": [1, 2, 3, 4],
        "dim_feedforward": [16, 32],
    }

    keys = list(search_space.keys())

    combinations = list(itertools.product(*(search_space[k] for k in keys)))
    results = []

    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))

        if params["d_model"] % params["nhead"] != 0:
            continue

        print(params)

        model_cfg = {
            "encoder": copy.deepcopy(
                cfg["training"]["encoder"]
            ),
            "num_layers": params["num_layers"]
        }

        model_cfg["encoder"].update({
            "d_model": params["d_model"],
            "nhead": params["nhead"],
            "dim_feedforward": params["dim_feedforward"]
        })

        run_name = f"experimentz_{i + 1:03d}"
        metrics = run_training(model_cfg=model_cfg, run_name=run_name)

        row = {
            **params,
            "parameters": metrics["parameters"],
        }

        for r, ler in metrics["lers"].items():
            row[f"ler_r{r}"] = ler

        for r, loss in metrics["losses"].items():
            row[f"loss_r{r}"] = loss

        results.append(row)

        with open("sweep_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=results[0].keys()
            )
            writer.writeheader()
            writer.writerows(results)

    print("\nHyperparameter sweep complete!")


if __name__ == "__main__":
    main()