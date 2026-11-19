"""Train the transformer decoder on a generate_data.py run.

Run from anywhere:
  python code/pipeline/train.py                          a new run (settings: CONFIG in config.py); tests.py runs first
  python code/pipeline/train.py --resume <run folder>    continue a stopped run from its last finished epoch
  python code/pipeline/train.py --skip-tests             start without running tests.py
Output: code/pipeline/train_runs/<YYYY-MM-DD_HHMMSS>/ with
  train.log      learning rate, train loss, quick val checks, full val after every epoch, all next to always-0 and MWPM
  history.json   every logged number, for plots
  last.pt        the state after the last finished epoch (what --resume continues from)
  best.pt        the weights of the epoch with the lowest val LER
  model.pt       the kept model = best.pt's weights + config, data run, token features: everything to rebuild the inputs
Val LER = mean over the val round counts (3-30) of each round count's logical error rate, so every r counts the same.
"""
import argparse
import copy
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch, torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from config import DATA_DIR, TRAIN_DIR, check_config, load_config
from stim_preprocessing import read_manifest
from training_data import DEVICE, epoch_batches, find_run, first_shots, longest_rounds, plan_epoch, prepare_data


LOG_TO_SCREEN = True           # False: train.log only (tests.py turns the screen off for its short runs)


def setup_logger(log_path, append: bool = False) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    close_logger(log)           # avoid duplicate lines if run_training() is called twice in one process
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    handlers = [logging.FileHandler(log_path, mode="a" if append else "w")]
    if LOG_TO_SCREEN:
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def close_logger(log: logging.Logger) -> None:
    # close train.log (on Windows an open file cannot be moved or deleted)
    for handler in log.handlers:
        handler.close()
    log.handlers.clear()


# Decoder is the attention pipeline e2e
class Decoder(nn.Module):
    def __init__(self, n_features: int, t: dict):
        super().__init__()
        d = t["encoder"]["d_model"]
        # embed into d_model dimensions. n_features = numbers per token, read from the data (6 by default)
        self.embed = nn.Linear(n_features, d)
        # self-attention and MLP, pre-LN (norm_first in the config). Pre-LN leaves the last layer's output
        # un-normalized, so the encoder gets a final LayerNorm (norm=). Built here, so every Decoder has its own weights.
        # enable_nested_tensor=False: that fast path does not work with norm_first and only helps padded batches
        layer = nn.TransformerEncoderLayer(**t["encoder"])
        self.encoder = nn.TransformerEncoder(layer, num_layers=t["num_layers"], norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        # "cls": one learned token is put in front of every sequence and its output is the summary g
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, d)) if t["pooling"] == "cls" else None
        # pooling z = w^Tg + b is single logit for single memory experiment (memory Z). sigmoid(z) = P(observable flipped)
        self.head = nn.Linear(d, 1)

    def forward(self, v):
        x = self.embed(v)
        if self.cls is not None:
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1)
        x = self.encoder(x)
        g = x[:, 0] if self.cls is not None else x.mean(1)   # "mean": g = 1/N * sum over tokens
        return self.head(g).squeeze(-1)


def make_scheduler(opt, t: dict, steps: int):
    # warmup over warmup_ratio of all steps, then linear or cosine decay over the rest
    warmup_steps = max(1, int(t["warmup_ratio"] * steps))
    decay_steps = steps - warmup_steps
    warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    if t["lr_scheduler"] == "cosine":
        decay = CosineAnnealingLR(opt, T_max=decay_steps)
    elif t["lr_scheduler"] == "linear":
        decay = LinearLR(opt, start_factor=1.0, end_factor=1e-3, total_iters=decay_steps)
    else:
        raise ValueError(f"unknown lr_scheduler: {t['lr_scheduler']!r}")
    return SequentialLR(opt, [warmup, decay], milestones=[warmup_steps])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)        # same seed -> same starting weights and dropout masks (also seeds every GPU)


def get_elements(t: dict, n_features: int, steps: int):
    """Model, optimizer, scheduler and loss for one run. steps = total optimizer steps, for the scheduler."""
    seed_everything(t["seed"])
    model = Decoder(n_features, t).to(DEVICE)
    # use AdamW as default
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    return model, opt, make_scheduler(opt, t, steps), nn.BCEWithLogitsLoss()


# ======================= evaluation ============================
# A model that always answers 0 already gets 98% right at r=3, so every model number is logged next to
# two references on the same shots: always-0 (the floor to beat) and PyMatching / MWPM (the target).

def baselines(run_dir: Path, split: str) -> dict[int, dict]:
    """Shots and errors of always-0 and MWPM per round count of a split, as generate_data.py counted them."""
    info = read_manifest(str(run_dir))["splits"][split]
    return {int(r): {"shots": s["shots"], "always0": s["always0_errors"], "mwpm": s.get("mwpm_errors")}
            for r, s in info.items()}


def check_baselines(run_dir: Path, part: dict[int, dict]) -> dict[int, dict]:
    """The same for the quick-check shots (the first few of each round count), decoded here with PyMatching."""
    import pymatching, stim
    out = {}
    for r, p in part.items():
        dem = stim.DetectorErrorModel.from_file(run_dir / "circuits" / f"r{r:03d}.dem")
        dets, labels = p["dets"].cpu().numpy(), p["labels"].cpu().numpy()
        guess = pymatching.Matching.from_detector_error_model(dem).decode_batch(dets)[:, 0]
        out[r] = {"shots": len(labels), "always0": int(labels.sum()), "mwpm": int((guess != labels).sum())}
    return out


@torch.no_grad()
def evaluate(model: Decoder, part: dict, cfg: dict, rounds) -> dict[int, dict]:
    """Model errors, mean BCE and number of shots predicted 1, per round count, over every given shot of those rounds."""
    was_training = model.training
    model.eval()
    out = {r: {"errors": 0, "ones": 0, "bce": 0.0} for r in rounds}
    for v, y, r in epoch_batches({r: part[r] for r in rounds}, cfg, 0, train=False):
        logits = model(v)
        pred = logits > 0   # logit > 0 is the same as sigmoid(logit) > 0.5
        out[r]["errors"] += int((pred != (y > 0.5)).sum())
        out[r]["ones"] += int(pred.sum())
        out[r]["bce"] += float(nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum"))
    for r in rounds:
        out[r]["bce"] /= len(part[r]["labels"])
    model.train(was_training)
    return out


def mean_ler(errors: dict[int, int], base: dict[int, dict]) -> float:
    # the val LER used to keep the best epoch and to compare runs: every round count weighs the same
    return sum(errors[r] / base[r]["shots"] for r in errors) / len(errors)


def per_round_error(ler: float, r: int) -> float:
    # LER after r rounds -> error per round: 1 - 2 * LER = (1 - 2 * eps)^r, so eps = (1 - (1 - 2 LER)^(1/r)) / 2
    return 0.5 * (1 - max(1 - 2 * ler, 0.0) ** (1 / r))


def pct(errors, shots) -> str:
    return "  n/a " if errors is None else f"{100 * errors / shots:5.2f}%"


def log_eval(log: logging.Logger, ev: dict, base: dict) -> None:
    for r, e in ev.items():
        b = base[r]
        log.info("  r=%3d | LER %s (%d of %d) | always-0 %s | MWPM %s (%s) | BCE %.4f", r, pct(e["errors"], b["shots"]),
                 e["errors"], b["shots"], pct(b["always0"], b["shots"]), pct(b["mwpm"], b["shots"]), b["mwpm"], e["bce"])
        if e["ones"] in (0, b["shots"]):
            log.warning("  r=%3d | model predicts %d on every shot: a constant, not a decoder", r, int(e["ones"] > 0))


def log_summary(log: logging.Logger, ev: dict, base: dict, title: str, cut: int) -> float:
    errors = {r: e["errors"] for r, e in ev.items()}
    val = mean_ler(errors, base)
    for name, rs in ((f"r <= {cut}", [r for r in ev if r <= cut]), (f"r > {cut}", [r for r in ev if r > cut])):
        shots = sum(base[r]["shots"] for r in rs)
        log.info("  %-8s | LER %s (%d of %d) | always-0 %s | MWPM %s", name, pct(sum(errors[r] for r in rs), shots),
                 sum(errors[r] for r in rs), shots, pct(sum(base[r]["always0"] for r in rs), shots),
                 pct(sum(base[r]["mwpm"] for r in rs), shots))
    ref = {k: mean_ler({r: base[r][k] for r in ev}, base) for k in ("always0", "mwpm")}
    log.info("%s | val LER (mean over r %d-%d) %.3f%% | always-0 %.3f%% | MWPM %.3f%%", title, min(ev), max(ev),
             100 * val, 100 * ref["always0"], 100 * ref["mwpm"])
    constant = [r for r, e in ev.items() if e["ones"] in (0, base[r]["shots"])]
    if constant:
        log.warning("model predicts the same answer on every shot of r = %s", constant)
    return val


def save_checkpoint(path, model: Decoder, opt, scheduler, epoch: int, step: int, best: dict, history: dict, cfg: dict):
    torch.save({
        "config": cfg,
        "epoch": epoch,             # epochs finished
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": scheduler.state_dict(),
        "best": best,
        "history": history,
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def peak_ram_gb() -> Optional[float]:
    try:
        import resource                                               # Linux / macOS
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 2**20 if sys.platform != "darwin" else kb / 2**30
    except ImportError:
        try:
            import psutil                                             # Windows, if psutil is installed
            return psutil.Process().memory_info().peak_wset / 2**30
        except (ImportError, AttributeError):
            return None


# ======================= training loop ============================

def run_training(cfg: Optional[dict] = None, data_dir: Path = DATA_DIR, out_dir: Path = TRAIN_DIR,
                 resume: Optional[Path] = None, stop_after: Optional[int] = None) -> dict:
    """Train one model and return its history. resume = a run folder to continue from its last.pt.
    stop_after = finish this many epochs in total and return (the rest can be resumed later)."""
    try:
        return _run_training(cfg, data_dir, out_dir, resume, stop_after)
    finally:
        close_logger(logging.getLogger("train"))   # also after an error: Windows cannot delete an open train.log


def _run_training(cfg: Optional[dict], data_dir: Path, out_dir: Path, resume: Optional[Path],
                  stop_after: Optional[int]) -> dict:
    if resume is not None:
        save_path = Path(resume)
        # on the CPU: the saved RNG states must stay CPU tensors; load_state_dict moves the rest to the model's device
        state = torch.load(save_path / "last.pt", map_location="cpu", weights_only=False)
        cfg = state["config"]   # the run continues with the settings it started with
        log = setup_logger(save_path / "train.log", append=True)
        log.info("\n===\nRESUMING %s after epoch %d (step %d)", save_path.name, state["epoch"], state["step"])
    else:
        cfg = copy.deepcopy(load_config() if cfg is None else cfg)
        cfg["training"]["run"] = find_run(cfg["training"]["run"], data_dir).name   # pin the data folder: --resume uses it too
        check_config(cfg)
        start_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_path = Path(out_dir) / start_time
        save_path.mkdir(parents=True, exist_ok=True)
        log = setup_logger(save_path / "train.log")
        log.info("\n===\nTRAINING RUN: %s", start_time)
        log.info("CONFIG: \n%s", json.dumps(cfg, indent=2))
    t = cfg["training"]
    E = t["num_epochs"]
    cut = cfg["rounds"]["short_bucket"][1]

    # load the syndromes (uint8) of train and val onto DEVICE once; tokens are built per batch
    data = prepare_data(cfg, data_dir)
    run_dir = data["run_dir"]
    train, val = data["train"], data["val"]
    log.info("DATA: %s | device %s | train %d shots, r %d-%d | val %d shots | token %s", run_dir, DEVICE,
             sum(len(p["labels"]) for p in train.values()), min(train), max(train),
             sum(len(p["labels"]) for p in val.values()), ", ".join(data["features"]))
    base = baselines(run_dir, "val")   # always-0 and MWPM never change: counted once, by generate_data.py
    check = first_shots(val, t["eval_rounds"], t["eval_check_shots"])
    check_base = check_baselines(run_dir, check)

    # total steps for the scheduler = batches in every epoch (the curriculum makes the first epochs shorter)
    batches_per_epoch = [len(plan_epoch(train, cfg, e, train=True)) for e in range(E)]
    total_steps = sum(batches_per_epoch)
    model, opt, scheduler, loss_fn = get_elements(t, len(data["features"]), total_steps)
    log.info("MODEL: %d parameters | %d steps (%s per epoch)", sum(p.numel() for p in model.parameters()),
             total_steps, batches_per_epoch)

    history = {"step": [], "lr": [], "train_loss": [], "val": {r: [] for r in check}, "epochs": []}
    best = {"val_ler": float("inf"), "epoch": 0}
    start_epoch, step = 0, 0
    if resume is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, step, best, history = state["epoch"], state["step"], state["best"], state["history"]
        history["val"] = {int(r): v for r, v in history["val"].items()}
    else:
        log.info("step 0 (before training), quick check on the first %d val shots of r = %s:", t["eval_check_shots"],
                 t["eval_rounds"])
        log_eval(log, evaluate(model, check, cfg, list(check)), check_base)

    losses = []
    last_epoch = E if stop_after is None else min(E, stop_after)
    for epoch in range(start_epoch, last_epoch):
        limit = longest_rounds(t, epoch)   # curriculum: longest r this epoch
        used = [r for r in train if limit is None or r <= limit]
        log.info("Beginning epoch %d of %d (%d steps, r %d-%d)", epoch + 1, E, batches_per_epoch[epoch], min(used), max(used))
        model.train()
        for Xt, yt, r in epoch_batches(train, cfg, epoch, train=True):
            loss = loss_fn(model(Xt), yt)
            opt.zero_grad()
            loss.backward()
            # rescale gradients so their total norm is at most grad_clip
            nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()         # update weights
            scheduler.step()   # update lr
            step += 1
            losses.append(loss.item())

            # every eval_every steps: mean train loss since the last check, then the quick val check
            if step % t["eval_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                mean_loss = sum(losses) / len(losses)
                losses = []
                log.info("step %d | epoch %d | lr %.3g | train loss %.4g (mean of last %d steps)",
                         step, epoch + 1, lr, mean_loss, t["eval_every"])
                ev = evaluate(model, check, cfg, list(check))
                log_eval(log, ev, check_base)
                history["step"].append(step)
                history["lr"].append(lr)
                history["train_loss"].append(mean_loss)
                for r in check:
                    history["val"][r].append(ev[r]["errors"] / check_base[r]["shots"])

        # end of epoch: every val shot of every round count; the epoch with the lowest val LER is kept
        ev = evaluate(model, val, cfg, sorted(val))
        val_ler = log_summary(log, ev, base, f"end of epoch {epoch + 1}", cut)
        history["epochs"].append({"epoch": epoch + 1, "step": step, "val_ler": val_ler,
                                  "per_round": {r: e["errors"] for r, e in ev.items()}})
        if val_ler < best["val_ler"]:
            best = {"val_ler": val_ler, "epoch": epoch + 1}
            torch.save(model.state_dict(), save_path / "best.pt")
            log.info("new best val LER: epoch %d", epoch + 1)
        save_checkpoint(save_path / "last.pt", model, opt, scheduler, epoch + 1, step, best, history, cfg)

    history["best"] = best
    if last_epoch < E:
        log.info("STOPPED after epoch %d of %d: continue with --resume %s", last_epoch, E, save_path)
    else:
        model.load_state_dict(torch.load(save_path / "best.pt", map_location=DEVICE, weights_only=True))
        model_path = save_path / "model.pt"
        log.info("TRAINING FINISHED | best val LER %.3f%% at epoch %d", 100 * best["val_ler"], best["epoch"])
        log.info("Saving the best epoch's model to %s", model_path)
        # everything needed to rebuild the inputs and the model later, not just the weights
        torch.save({
            "config": cfg,
            "data_run": str(run_dir),
            "data_config": read_manifest(str(run_dir))["config"],
            "features": data["features"],
            "tokens": cfg["tokens"],    # time = round index / r (relative); sincos periods if used
            "epoch": best["epoch"],
            "val_ler": best["val_ler"],
            "model": model.state_dict(),
        }, model_path)
    ram = peak_ram_gb()
    log.info("peak RAM of this process: %s\n", "n/a (pip install psutil on Windows)" if ram is None else f"{ram:.2f} GB")
    (save_path / "history.json").write_text(json.dumps(history, indent=1))
    close_logger(log)
    history["folder"] = str(save_path)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", type=Path, help="a train_runs/<run> folder to continue")
    parser.add_argument("--skip-tests", action="store_true", help="do not run tests.py first")
    args = parser.parse_args()
    if not args.skip_tests and args.resume is None:
        import tests
        tests.run_all()
    run_training(resume=args.resume)
