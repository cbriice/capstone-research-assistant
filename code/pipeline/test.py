"""
hand written and maintained testing script. includes MWPM baseline for same dataset as was trained on
hardcoded:
- training and dataset paths    :17-18
- path to save plot              :72
"""
import torch, json
from pipeline.train import Decoder
from pipeline.stim_preprocessing import load_tokens
from pipeline.config import load_config
import matplotlib.pyplot as plt
from pathlib import Path
import stim
import numpy as np
from pymatching import Matching

def test():
    DATASET = Path("pipeline/data/2026-11-19_180037")
    MODEL_PATH = "pipeline/data/train/2026-11-19_182540/model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print(f"Testing {MODEL_PATH}")

    train_cfg = json.loads(checkpoint["config"])
    # pull current config and match. if current config not the same as saved, then model.load_state_dict() fails because Decoder() pulls module-level config
    current_cfg = json.loads(json.dumps(load_config()["training"], default=str))
    arch_keys = ["encoder", "num_layers"]   # the keys that determine architecture / state_dict shapes
    for k in arch_keys:
        assert train_cfg[k] == current_cfg[k], f"{k} mismatch:\n saved:   {train_cfg[k]}\n current: {current_cfg[k]}"

    # get rounds from config
    cfg = load_config()
    r_low, r_hi = cfg["rounds"]
    model = Decoder().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lers = []
    mwpm_lers = []
    ns = []
    for r in range(r_low, r_hi + 1):
        # transformer test
        X, y = load_tokens(DATASET, "test", r)
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            logits = torch.cat([model(xc) for xc in X.split(10000)])
            preds = (logits > 0).float()
            ler = (preds != y).float().mean() 
            lers.append(ler.item())
        ns.append(len(y))
        del X, y
        # MWPM comparison
        dem = stim.DetectorErrorModel.from_file(DATASET / "circuits" / f"r{r:03d}.dem")
        f = np.load(DATASET / "test" / f"r{r:03d}.npz")
        pred = Matching.from_detector_error_model(dem).decode_batch(f["dets"])
        mwpm_ler = (pred[:, 0] != f["obs"][:, 0]).mean()
        del f
        mwpm_lers.append(mwpm_ler)
        print(f"r={r} | TRANSFORMER={ler:.4f}, MWPM={mwpm_ler:.4f}")

    # function to take standard error of logical error rate over n shots. p=LER n=num_shots at given r
    def se(p, n):
        p, n = np.asarray(p), np.asarray(n)
        return np.sqrt(p * (1 - p) / n)
    
    fig, ax = plt.subplots()
    ax.errorbar(range(r_low, r_hi+1), lers, yerr=se(lers, ns), capsize=3, label="Transformer")
    ax.errorbar(range(r_low, r_hi+1), mwpm_lers, yerr=se(mwpm_lers, ns), capsize=3, label="MWPM", color='r')
    ax.legend()
    plt.title("Z memory LERs for each round count 3-10 at p=1e-3 uniform stochastic pauli noise")
    plt.xlabel("rounds")
    plt.ylabel("LER")
    path = Path("train_plots/2026-11-19_183646/test.png")
    fig.savefig(path)
    print(f"Saved plot to {path}")

if __name__ == "__main__":
    test()
