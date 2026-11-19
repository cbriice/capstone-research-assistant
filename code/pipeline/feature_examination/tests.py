"""Quick checks for the pipeline (under a minute on a CPU, writes only to a temporary folder).
train.py and experiments.py run these first, so a broken change stops before hours of training.

Run from anywhere:  python code/pipeline/tests.py
  1. test_your_config:    config.py as it is now passes the checks, and one batch of your tokens goes through your
                          model forwards and backwards
  Tests 2-6 check the code with the settings config.py ships with (SHIPPED below), so they pass whatever you change
  in CONFIG (features, model size, rounds, curriculum ...):
  2. test_config:         the shipped settings pass the checks, and a wrong value is an error
  3. test_lr_schedule:    the learning rate is right at the key steps of both schedules
  4. test_token_features: every feature of a tiny circuit has the value it should, on CPU and in the GPU batches
  5. test_batches:        curriculum, mixed and unmixed batch orders, every shot once per epoch
  6. test_smoke_run:      tiny data set -> sizes and baselines in the manifest -> a short training run that beats
                          always-guess-0 -> a stopped and resumed run that ends with exactly the same weights ->
                          evaluate.py on the test split
"""
import copy
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import evaluate
import generate_data
import train
from config import check_config, load_config, test_rounds, train_rounds
from stim_preprocessing import make_tokens, static_features
from training_data import DEVICE, make_batch, plan_epoch, prepare_data, token_names

SHIPPED = {                                                                                     #the settings config.py ships with
    "circuit": {"distance": 3, "basis": "z", "p": 0.001},
    "rounds": {"short_bucket": [3, 15], "long_bucket": [16, 30], "long_fraction": 0.2, "test_max": 50},
    "dataset": {"train_shots": 900_000, "train_empty_keep_fraction": 1 / 3, "eval_mwpm_errors": 150,
                "eval_min_shots": 2_000, "eval_max_shots": 300_000, "pilot_shots": 100_000},
    "tokens": {"features": ["x", "y", "t", "is_z", "is_final"], "sincos_periods": [8, 64]},
    "training": {"run": None, "seed": 0,
                 "encoder": {"d_model": 16, "nhead": 1, "dim_feedforward": 64, "dropout": 0.05, "activation": "gelu",
                             "batch_first": True, "norm_first": True},
                 "num_layers": 2, "pooling": "mean", "batch_size": 256, "max_tokens_per_batch": 65_536,
                 "curriculum": [5, 10, 20], "mix_rounds": True, "warmup_ratio": 0.1, "lr_scheduler": "linear",
                 "num_epochs": 10, "lr": 2e-3, "weight_decay": 0.01, "grad_clip": 1.0, "eval_every": 100,
                 "eval_rounds": [3, 10, 20, 30], "eval_check_shots": 2_000},
    "evaluation": {"target_x_mwpm": 2.0}}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1e-12)


def shipped_config() -> dict:                                                                   #CONFIG with every value reset to SHIPPED
    cfg = load_config()
    for block, values in SHIPPED.items():
        cfg[block].update(copy.deepcopy(values))                                                #keeps keys SHIPPED does not list
    check_config(cfg)
    return cfg


def must_fail(cfg: dict, why: str) -> None:
    try:
        check_config(cfg)
    except ValueError:
        return
    raise AssertionError(f"check_config accepted {why}")


def test_your_config() -> None:
    cfg = load_config()                                                                         #config.py as it is now; load_config checks it
    c, k, t = cfg["circuit"], cfg["tokens"], cfg["training"]
    names = token_names(k)
    circuit = generate_data.make_circuit(c["distance"], 3, c["p"], c["basis"])
    dets = np.zeros((4, circuit.num_detectors), dtype=bool)
    dets[1, [0, 5]] = True
    x = torch.from_numpy(make_tokens(circuit, dets, c["basis"], k["features"], k["sincos_periods"]))
    assert x.shape == (4, circuit.num_detectors, len(names)), x.shape
    model = train.Decoder(len(names), t)
    logits = model(x)
    assert logits.shape == (4,) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros(4)).backward()     #one step's worth of gradients
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    params = sum(p.numel() for p in model.parameters())
    e = t["encoder"]
    print(f"your config: token {', '.join(names)} ({len(names)} numbers) | d_model {e['d_model']}, {e['nhead']} head(s), "
          f"ff {e['dim_feedforward']}, {t['num_layers']} layers = {params:,} parameters | lr {t['lr']:g}")
    longest = max(train_rounds(cfg))
    tokens = generate_data.make_circuit(c["distance"], longest, c["p"], c["basis"]).num_detectors
    if t["batch_size"] * tokens > t["max_tokens_per_batch"]:                                    #allowed, but worth knowing
        print(f"note: at r = {longest} a batch of {t['batch_size']} shots is {t['batch_size'] * tokens:,} tokens, over "
              f"max_tokens_per_batch {t['max_tokens_per_batch']:,}, so long-r batches hold fewer shots")


def test_config() -> None:
    cfg = shipped_config()
    assert load_config() is not load_config()                                                   #load_config returns a copy
    assert train_rounds(cfg) == list(range(3, 31)) and test_rounds(cfg) == list(range(3, 51))
    t = cfg["training"]
    assert t["batch_size"] * 8 * 30 <= t["max_tokens_per_batch"], "the token cap would bind below r = 30"
    for change, why in ((("tokens", "features", ["y", "x"]), "features out of order"),
                        (("tokens", "features", ["z"]), "an unknown feature"),
                        (("training", "curriculum", [5, 40]), "a curriculum beyond the training rounds"),
                        (("rounds", "long_bucket", [15, 30]), "overlapping buckets"),
                        (("rounds", "test_max", 20), "a test range shorter than the training range")):
        bad = copy.deepcopy(cfg)
        bad[change[0]][change[1]] = change[2]
        must_fail(bad, why)


def test_lr_schedule() -> None:
    t = shipped_config()["training"]
    lr, steps = t["lr"], 1000                                                                   #warmup_ratio 0.1 -> 100 warmup steps
    for kind, middle in (("linear", lr * (1 - 0.999 * 0.5)), ("cosine", lr * 0.5)):             #lr halfway through the decay
        t["lr_scheduler"] = kind
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)
        sched = train.make_scheduler(opt, t, steps)
        lrs = []
        for _ in range(steps):
            lrs.append(opt.param_groups[0]["lr"])                                               #lr used by this step
            opt.step()
            sched.step()
        assert close(lrs[0], lr * 1e-3), (kind, lrs[0])                                         #warmup starts at lr / 1000
        assert close(lrs[100], lr) and close(max(lrs), lr), (kind, lrs[100])                    #peak lr right after warmup
        assert all(a <= b for a, b in zip(lrs[:100], lrs[1:101])), kind                         #rises during warmup
        assert all(a >= b for a, b in zip(lrs[100:], lrs[101:])), kind                          #falls after it
        assert close(lrs[550], middle), (kind, lrs[550], middle)                                #halfway: 100 + 900 / 2
        assert lrs[-1] < 0.01 * lr, (kind, lrs[-1])                                             #near zero at the end, not stuck at step 6
    if kind == "linear":
        assert close(opt.param_groups[0]["lr"], lr * 1e-3)                                      #after the last step: lr / 1000


def test_token_features() -> None:
    circuit = generate_data.make_circuit(distance=3, rounds=2, p=0.001, basis="z")              #16 detectors: 4 at t=0, 8 at t=1, 4 at t=2
    dets = np.zeros((2, circuit.num_detectors), dtype=bool)
    dets[1, [0, 5, 15]] = True                                                                  #shot 1: three detectors fired
    every = ["x", "y", "t", "is_z", "is_final", "is_first", "sincos"]
    tokens = make_tokens(circuit, dets, "z", every, [4, 64])
    assert tokens.shape == (2, 16, 1 + 6 + 4) and tokens.dtype == np.float32
    event, x, y, t, is_z, is_final, is_first, s4, c4, s64, c64 = tokens[1].T
    assert (event == dets[1]).all()
    assert (tokens[0, :, 0] == 0).all() and (tokens[0, :, 1:] == tokens[1, :, 1:]).all()        #only the event differs
    assert sorted(set(t)) == [0, 0.5, 1] and list(t).count(0) == 4 and list(t).count(1) == 4    #round index / r
    assert x.min() >= -0.5 and x.max() <= 0.5 and y.min() >= -0.5 and y.max() <= 0.5
    assert len(set(zip(x, y))) == 8                                                             #8 stabilizers at d=3
    assert is_z.sum() == 12 and is_z[t == 0].all() and is_z[t == 0.5].sum() == 4                #z memory: only Z checks at t=0 and t=2
    assert (is_final == (t == 1)).all() and (is_first == (t == 0)).all()
    k = 2 * t                                                                                   #absolute round index 0, 1, 2
    assert np.allclose(s4, np.sin(2 * np.pi * k / 4), atol=1e-6) and np.allclose(c4, np.cos(np.pi * k / 2), atol=1e-6)
    assert np.allclose(s64, np.sin(2 * np.pi * k / 64), atol=1e-6) and np.allclose(c64, np.cos(2 * np.pi * k / 64), atol=1e-6)
    sub = make_tokens(circuit, dets, "z", ["t", "is_final"], [4, 64])                           #a subset keeps the token order
    assert np.array_equal(sub, tokens[:, :, [0, 3, 5]])
    default = make_tokens(circuit, dets, "z")
    assert np.array_equal(default, tokens[:, :, :6])                                            #default: event, x, y, t, is_z, is_final
    r4 = make_tokens(generate_data.make_circuit(3, 4, 0.001, "z"), np.zeros((1, 32), bool), "z", ["t", "sincos"], [8, 64])
    r100 = make_tokens(generate_data.make_circuit(3, 100, 0.001, "z"), np.zeros((1, 800), bool), "z", ["t", "sincos"], [8, 64])
    a, b = r4[0][r4[0, :, 1] == 0.75][0], r100[0][r100[0, :, 1] == 0.75][0]                     #round 3 of 4 and round 75 of 100
    assert a[1] == b[1] and not np.allclose(a[2:], b[2:])                                       #same t / r, different sin / cos
    x_basis = make_tokens(generate_data.make_circuit(3, 2, 0.001, "x"), dets, "x")[1, :, 4]
    assert x_basis.sum() == 4                                                                   #x memory: the 4 Z checks are only in the middle slice
    part = {"dets": torch.from_numpy(dets.astype(np.uint8)), "labels": torch.tensor([0.0, 1.0]),
            "static": torch.from_numpy(static_features(circuit, "z", every, [4, 64]))}
    x_gpu, y_gpu = make_batch(part, torch.tensor([1, 0]))                                       #the batch builder train.py uses
    assert torch.equal(x_gpu, torch.from_numpy(tokens[[1, 0]])) and torch.equal(y_gpu, torch.tensor([1.0, 0.0]))
    t_cfg = shipped_config()["training"]
    a, b = train.Decoder(6, t_cfg), train.Decoder(6, t_cfg)
    assert a.encoder is not b.encoder                                                           #each model has its own encoder
    assert a.encoder.layers[0].norm_first and isinstance(a.encoder.norm, torch.nn.LayerNorm)    #pre-LN + final LayerNorm
    assert close(train.per_round_error(0.5 * (1 - 0.98 ** 10), 10), 0.01)                       #eps = 1% per round over 10 rounds


def fake_part(rounds: dict[int, int]) -> dict[int, dict]:
    return {r: {"dets": torch.zeros(n, 8 * r, dtype=torch.uint8), "labels": torch.zeros(n)} for r, n in rounds.items()}


def test_batches() -> None:
    cfg = shipped_config()
    t = cfg["training"]
    part = fake_part({r: 1000 for r in range(3, 31)})
    for epoch, longest in ((0, 5), (1, 10), (2, 20), (3, 30), (9, 30)):                         #curriculum 5, 10, 20, then every r
        plan = plan_epoch(part, cfg, epoch, train=True)
        rounds = {r for r, _ in plan}
        assert max(rounds) == longest and min(rounds) == 3, (epoch, sorted(rounds))
        for r in rounds:
            idx = np.concatenate([i for rr, i in plan if rr == r])
            assert sorted(idx) == list(range(1000)), r                                          #every shot of an included r, once
        assert all(len(i) <= t["batch_size"] for _, i in plan)
    assert plan_epoch(part, cfg, 4, True) != plan_epoch(part, cfg, 5, True)                     #a new order every epoch
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(plan_epoch(part, cfg, 4, True), plan_epoch(part, cfg, 4, True)))
    t["mix_rounds"] = False
    rounds = [r for r, _ in plan_epoch(part, cfg, 5, train=True)]
    blocks = [r for i, r in enumerate(rounds) if not i or r != rounds[i - 1]]                   #r of each run of equal r
    assert sorted(blocks) == list(range(3, 31)) and blocks != sorted(blocks), blocks            #unmixed: each r once, shuffled order


SMOKE_P, SMOKE_SHOTS, SMOKE_BATCH, SMOKE_LR = 0.003, 60000, 128, 2e-3


def smoke_config() -> dict:
    cfg = shipped_config()
    cfg["circuit"]["p"] = SMOKE_P
    cfg["rounds"] = {"short_bucket": [3, 5], "long_bucket": [6, 7], "long_fraction": 0.2, "test_max": 9}
    cfg["dataset"].update(train_shots=SMOKE_SHOTS, eval_mwpm_errors=40, eval_min_shots=500, eval_max_shots=3000,
                          pilot_shots=5000)
    t = cfg["training"]
    t.update(run=None, num_epochs=2, batch_size=SMOKE_BATCH, curriculum=[5], eval_every=150, eval_rounds=[3, 7],
             eval_check_shots=300, lr=SMOKE_LR)
    check_config(cfg)
    return cfg


def test_smoke_run() -> None:
    cfg = smoke_config()
    train.LOG_TO_SCREEN = False                                                                 #train.log is still written; keeps this output short
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:                        #Windows: a locked file must not hide the real error
        tmp = Path(tmp)
        run_dir = generate_data.main(cfg, data_dir=tmp / "data")
        base = train.baselines(run_dir, "test")
        assert sorted(base) == list(range(3, 10)) and sorted(train.baselines(run_dir, "val")) == list(range(3, 8))
        assert all(500 <= b["shots"] <= 3000 and b["mwpm"] is not None for b in base.values())
        data = prepare_data(cfg, tmp / "data")
        assert sum(len(p["labels"]) for p in data["train"].values()) == SMOKE_SHOTS
        assert data["features"] == ("event", "x", "y", "t", "is_z", "is_final")

        history = train.run_training(cfg, data_dir=tmp / "data", out_dir=tmp / "runs")          #one run straight through
        folder = Path(history["folder"])
        assert sorted(p.name for p in folder.iterdir()) == ["best.pt", "history.json", "last.pt", "model.pt", "train.log"]
        saved = torch.load(folder / "model.pt", weights_only=False)
        pinned = {**cfg, "training": {**cfg["training"], "run": run_dir.name}}                  #run None became the data folder used
        assert saved["config"] == pinned and saved["features"] == data["features"]
        assert saved["epoch"] == history["best"]["epoch"]
        vals = [e["val_ler"] for e in history["epochs"]]
        assert close(saved["val_ler"], min(vals))                                               #the kept model is the best epoch

        stopped = train.run_training(cfg, data_dir=tmp / "data", out_dir=tmp / "runs2", stop_after=1)
        resumed = train.run_training(data_dir=tmp / "data", resume=Path(stopped["folder"]))     #stop after epoch 1, then resume
        a = torch.load(folder / "last.pt", weights_only=False)["model"]
        b = torch.load(Path(resumed["folder"]) / "last.pt", weights_only=False)["model"]
        assert len(resumed["epochs"]) == len(vals) and Path(resumed["folder"], "model.pt").exists()
        same = all(torch.equal(a[k], b[k]) for k in a) and [e["val_ler"] for e in resumed["epochs"]] == vals
        if DEVICE == "cpu":
            assert same, "a resumed run must end with the same weights"
        elif not same:                                                                          #CUDA kernels are not bit-for-bit repeatable
            print("note: on the GPU the resumed run is close to, not bit-identical with, the straight run (normal for CUDA)")

        result = evaluate.main(folder, data_dir=tmp / "data")
        assert [x["r"] for x in result["per_round"]] == list(range(3, 10))
        assert all((folder / f).exists() for f in ("test_results.csv", "test_ler.png", "test_per_round.png"))
        try:
            evaluate.main(folder, data_dir=tmp / "data")
            raise AssertionError("testing twice must need --again")
        except SystemExit:
            pass
    train.LOG_TO_SCREEN = True
    first = history["epochs"][-1]
    always0 = sum(x["always0"] for x in result["per_round"]) / len(result["per_round"])
    model = sum(x["ler"] for x in result["per_round"]) / len(result["per_round"])
    print(f"smoke run: val LER {100 * first['val_ler']:.2f}% | test LER {100 * model:.2f}% | always-0 {100 * always0:.2f}%")
    assert model < 0.8 * always0, "the model did not beat always-guess-0 by 20%"


def run_all() -> None:
    for test in (test_your_config, test_config, test_lr_schedule, test_token_features, test_batches, test_smoke_run):
        t0 = time.perf_counter()
        test()
        print(f"PASS {test.__name__} ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    run_all()
