import torch, torch.nn as nn
from pipeline.config import load_config
from pipeline.stim_preprocessing import load_tokens
import json, logging, random, time
from datetime import datetime
from pathlib import Path
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np

# each training run gets its own folder under WORK_DIR: checkpoints, final model, train.log
WORK_DIR = "pipeline/data/train/"

def setup_logger(log_path: str) -> logging.Logger:
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    log.handlers.clear()        # avoid duplicate lines if run_training() is called twice in one process
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log

# data directory
DATA_DIR = "pipeline/data/2026-11-17_144943/"

# initialize TransformerEncoder from config
cfg = load_config()
t = cfg["training"]
encoder_params = t["encoder"]
L = t["num_layers"]
LR = t["lr"]
E = t["num_epochs"]
B = t["batch_size"]

encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**encoder_params), num_layers=L)   # consider norm_first if training unstable

# Decoder is the attention pipeline e2e
class Decoder(nn.Module):
    def __init__(self, d=encoder_params["d_model"]):
        super().__init__()
        # embed into d_model dimensions. 6 = num features per token.
        self.embed = nn.Linear(6, d)
        # self-attention and MLP. default config: norm after included. normalized output fed to linear pooler
        self.encoder = encoder
        # pooling z = w^Tg + b is single logit for single memory experiment. V1 ONLY. in practice out_features=2; X and Z frame
        self.head = nn.Linear(d, 1)

    def forward(self, v): 
        x = self.encoder(self.embed(v))
        return self.head(x.mean(1)).squeeze(-1)

def get_elements(steps):
    """Get the shit you need for training. Pass steps as param for setting scheduler config"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Decoder().to(device)
    # use AdamW as default. maybe we change this idk its pretty standard
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=t["weight_decay"])
    # warmup + either cosine or linear depending on config. linear default is arbitrary choice
    warmup_steps = max(1, int(t["warmup_ratio"]*steps))
    if t["lr_scheduler"] == "cosine":
        warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(opt, T_max=steps - warmup_steps)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        linear = LinearLR(opt, start_factor=1.0, end_factor=1e-3, total_iters=steps - warmup_steps)
        scheduler = SequentialLR(opt, schedulers=[warmup, linear], milestones=[warmup_steps])

    loss_fn = nn.BCEWithLogitsLoss()

    return device, model, opt, scheduler, loss_fn

# define evaluate function to check progress during training. @torch.no_grad() blocks gradient computation to save memory since evaluate is an inference function
@torch.no_grad()
def evaluate(model: Decoder, loss_fn, v, y, log):
    model.eval()
    logits = model(v)
    predictors = (logits > 0).float()           # (model(v) > 0) converts logits to predictor values. equivalent to sigmoid(model(v)) > 0.5.
    ler = (predictors != y).float().mean()    
    loss = loss_fn(logits, y.float())           # hand-calculate loss here to track per fixed round
    if not predictors.any():
        log.warning("EVAL: Model predicted all zero")
    return ler.item(), loss.item(), predictors.mean(), y.mean()

def save_checkpoint(model: Decoder, opt, scheduler, path, step): 
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": scheduler.state_dict(),
        "rng": torch.get_rng_state(),
    }, path)

# ======================= training loop ============================

def run_training():
    start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_path = WORK_DIR + start_time
    Path(save_path).mkdir(parents=True, exist_ok=True)

    t_cfg = json.dumps(t, indent=2, default=str)
    log = setup_logger(save_path + "/train.log")
    log.info("\n===\nTRAINING RUN: %s", start_time)
    log.info("TRAINING CONFIG: \n%s", t_cfg)
            
    # global step counter
    num_steps = 0
    # load training data and unpack rounds in fixed batch sizes
    # assume working dir is code/ and data is fixed under pipeline/data/{day}/train
    r_min, r_max = cfg["rounds"]
    batches = {r: load_tokens(DATA_DIR, "train", r) for r in range(r_min, r_max + 1)}
    chunks = [(xs, ys) for X, y in batches.values() for xs, ys in zip(X.split(B), y.split(B))]
    total_steps = E * len(chunks)
    device, model, opt, scheduler, loss_fn = get_elements(total_steps)

    # load eval data
    lers = {}
    losses = {}
    eval_data = {}
    eval_rounds = [3, 10, 15, 25]
    for r in eval_rounds:
        lers[r] = []
        losses[r] = []
        Xv, yv = load_tokens(DATA_DIR, "val", r)
        Xv, yv = Xv.to(device), yv.to(device)
        eval_data[r] = (Xv, yv)

    # epoch outer loop
    for epoch in range(E):
        log.info("Beginning epoch %s", epoch+1)
        # shuffle the chunks to randomize train order
        random.shuffle(chunks)

        # inner loop over all round groups
        for i, (Xt, yt) in enumerate(chunks):
            model.train()
            Xt, yt = Xt.to(device), yt.to(device)
            loss = loss_fn(model(Xt), yt.float())
            opt.zero_grad()
            loss.backward()
            # normalize gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step() # update weights
            scheduler.step() # update lr

            # every 100 steps, run evaluate()
            if i % 100 == 0:
                log.info("entering eval at steps=%d", num_steps+i)
                for r in eval_rounds:
                    Xv, yv = eval_data[r]
                    ler, loss_eval, pred_mean, y_mean = evaluate(model, loss_fn, Xv, yv, log)
                    log.info("r= %d, LER=%.4g, loss=%.4g, predictor_mean=%.4g, true_mean=%0.4g", r, ler, loss_eval, pred_mean, y_mean)
                    if ler > 0.5:
                        log.warning("High LER detected")
                    lers[r].append(ler)
                    losses[r].append(loss_eval)
        # save checkpoint at each epoch
        num_steps += len(chunks)
        log.info("saving checkpoint for epoch %d", epoch+1)
        # don't need to save checkpoint for last epoch
        if epoch != 4: save_checkpoint(model, opt, scheduler, save_path + f"/e{epoch:02d}", num_steps)

    model_path = save_path + "/model"
    end = time.time()
    log.info(f"TRAINING FINISHED | elapsed: {end - start}s")
    log.info("Saving model to %s\n", model_path)
    torch.save({
        "config": t_cfg,
        "model": model.state_dict()
    }, model_path)
    # return data for analysis
    return lers, losses, total_steps
            
if __name__ == "__main__":
    run_training()