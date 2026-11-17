import torch, json
from pipeline.train import Decoder
from pipeline.stim_preprocessing import load_tokens
from pipeline.config import load_config
import matplotlib.pyplot as plt
from pathlib import Path

def test():
    MODEL_PATH = "pipeline/data/train/2026-11-17_145425/model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print(f"Testing {MODEL_PATH}")

    train_cfg = json.loads(checkpoint["config"])
    # pull current config and match. if current config not the same as saved, then model.load_state_dict() fails because Decoder() pulls module-level config
    current_cfg = json.loads(json.dumps(load_config()["training"], default=str))
    arch_keys = ["encoder", "num_layers"]   # the keys that determine architecture / state_dict shapes
    for k in arch_keys:
        assert train_cfg[k] == current_cfg[k], f"{k} mismatch:\n saved:   {train_cfg[k]}\n current: {current_cfg[k]}"

    model = Decoder().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    lers = []
    for r in range(3, 26):
        X, y = load_tokens("pipeline/data/2026-11-17_144943", "test", r)
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            logits = model(X)
            preds = (logits > 0).float()
            ler = (preds != y).float().mean() 
            lers.append(ler.item())
            print(f"TEST: r={r}, LER={ler:.4f}")

    fig, ax = plt.subplots()
    ax.plot(range(3, 26), lers)
    plt.title("Test: LERs for each round count 3-25")
    plt.xlabel("rounds")
    plt.ylabel("LER")
    fig.tight_layout()
    path = Path("train_plots/2026-11-17_145514/test.png")
    fig.savefig(path)
    print(f"Saved plot to {path}")

if __name__ == "__main__":
    test()
