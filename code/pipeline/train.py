import torch.nn as nn
import torch
from config import load_config
from stim_preprocessing import load_tokens
import random
import numpy as np

# initialize TransformerEncoder from config
cfg = load_config()
t = cfg["training"]
encoder_params = t["encoder"]
L = t["num_layers"]
LR = t["lr"]
E = t["num_epochs"]
B = t["batch_size"]
N = cfg["dataset"]["train_val_shots"]

encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**encoder_params), num_layers=L)   # consider norm_first if training unstable

# assume working dir is code/ and data is fixed under pipeline/data/{day}/train
# get data in batches by round
batches = {r: load_tokens("pipeline/data/2026-11-15_202422/", "train", r) for r in range(3, 101)}

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Decoder().to(device)
# use AdamW as default. maybe we change this idk its pretty standard
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=t["weight_decay"])
if t["lr_scheduler"] == "cosine":
    # arbitrarily choose 10*LR for maximum. could be changed
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=N)
else:
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=opt, end_factor=10)
loss_fn = nn.BCEWithLogitsLoss()

# define evaluate function to check progress during training. @torch.no_grad() blocks gradient computation to save memory since evaluate is an inference function
@torch.no_grad()
def evaluate(v, y):
    model.eval()
    return ((model(v) > 0) != y.bool()).float().mean() # LER

# ======================= training loop ============================

def run_training():
    # epoch outer loop
    for _ in range(E):
        # unpack rounds in fixed batch sizes and then shuffle the batches 
        chunks = []
        for X, y in batches.values():
            chunks.append((xs, ys) for xs, ys in zip(torch.split(X, B), torch.split(y, B)))
        # first inner loop over all round groups
        for Xt, yt in chunks:
            Xt = Xt.to(device)
            yt = yt.to(device)
            loss = loss_fn(model(Xt), yt.float())
            opt.zero_grad()
            loss.backward()
            # normalize gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step() # update weights
            scheduler.step() # update lr

            
if __name__ == "__main__":
    run_training()