# Machine-Learning Decoders — Study Path, Reproducible Results, and Project Ideas

*Author: Youngseok Kim*
*Last updated: 2026-08-25*

This note has three purposes, in order:

1. **What to read** to get up to speed on syndrome measurement and decoding (§1).
2. **Three papers where machine learning demonstrably improves decoding** — chosen because in each case *the algorithm is specified in enough detail and the data is public*, so the results are reproducible rather than merely admirable (§2).
3. **What you can build once you understand the data structure** (§3) — two concrete directions: a hardware compression front-end you could implement on an FPGA, and a leaner model architecture.

The through-line: a QEC decoder is, from a machine-learning point of view, a sequence model over a sparse binary tensor. Once that tensor is in hand — and it *is* in hand, publicly, from Google's Sycamore and Willow experiments — everything else is ordinary ML and hardware engineering.

---

## 0. Companion notes in this folder

Read these first or alongside; this note assumes them rather than repeating them.

| Note | What it gives you |
|---|---|
| `DEM.md` | What a detector error model is, how detector bits arise from syndrome measurement, and why DEM accuracy translates directly into logical error rate. **This is the definition of the decoder's input and prior.** |
| `DEM_prior.md` | Where a decoder's DEM prior `(p_e, h_e, l_e)` actually comes from — device noise model → Stim → parity rule (Path A) vs. empirical detector correlations from syndrome data (Path B). Worked example. |
| `DEM_prior_diagram.pdf` | The decoding chain as a picture. |
| `QED_vs_QEC_nonClifford_tradeoff.md` | Why detection is cheap and correction is expensive (Eastin–Knill). Context for *why* decoder quality is the bottleneck. |

The one-line map from `DEM_prior.md` is worth restating, because everything in §2 and §3 lives on this chain:

```
experiment → raw measurement outcomes → detector bits → decoder (+ DEM prior) → logical correction → logical error rate
                                            ^                    ^
                                            |                    |
                                     this is the ML input   this is what ML replaces
```

---

## 1. Background reading: syndrome measurement and decoders

### 1.1 Start here — the code and the syndrome-extraction circuit

- **Fowler, Mariantoni, Martinis, Cleland, *Surface codes: towards practical large-scale quantum computation*, Phys. Rev. A 86, 032324 (2012)** [^fowler2012]. Still the canonical tutorial. Read it for the stabilizer-measurement circuit, the space-time picture, and the matching-graph construction. Long, but §§I–V is enough.
- **Terhal, *Quantum error correction for quantum memories*, Rev. Mod. Phys. 87, 307 (2015)** [^terhal2015]. The review that puts surface codes in the broader QEC-memory context, including the backlog/throughput argument that later motivates real-time decoding.
- **Roffe, *Quantum error correction: an introductory guide*, Contemp. Phys. 60, 226 (2019)** [^roffe2019]. Shortest useful on-ramp if the stabilizer formalism is rusty.

### 1.2 The detector formalism — what the decoder actually consumes

This is the step most treatments skip, and it is exactly the step that matters for ML.

- **Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021)** [^stim]. Not just a tool paper: Stim's `DETECTOR` / `OBSERVABLE_INCLUDE` abstraction *is* the modern definition of a decoding problem. Learn `stim.Circuit`, `detector_error_model()`, and the sampler output formats (`.b8`, `.dets`, `.01`) — you will use all of them in §3.
- **Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*, Nature 614, 676 (2023)** [^google2023] and **_Quantum error correction below the surface code threshold_, Nature 638, 920 (2025)** [^google2025]. The two reference experiments. Read the Methods for how detection events, leakage, and soft (I/Q) readout are defined in practice.
- Then `DEM.md` and `DEM_prior.md` in this folder.

### 1.3 Decoder algorithms — the classical baselines you must beat

- **deMarti iOlius, Fuentes, Orús, Crespo, Etxezarreta Martinez, *Decoding algorithms for surface codes*, Quantum 8, 1498 (2024)** [^decodingreview]. **The single best survey.** MWPM, union-find, BP+OSD, tensor-network, and their complexity/accuracy trade-offs, in one place. If you read only one thing from §1, read this.
- **Higgott, *PyMatching*, arXiv:2105.13082 (2021)** and **Higgott & Gidney, *Sparse blossom*, Quantum 9, 1600 (2025)** [^pymatching]. The practical MWPM implementation — and your default baseline in any experiment.
- Named baselines you will keep seeing in the ML papers, worth knowing by reputation: **Tesseract** (A*-search, near-optimal, slow) [^tesseract], **Libra** (matching-synthesis ensembling, near-optimal) [^libra], **Chromobius** (fast colour-code decoder, lower accuracy) [^chromobius].

### 1.4 The real-time constraint

- **Battistel, Chamberland, Johar, Overwater, Sebastiano, Skoric, Ueno, Usman, *Real-time decoding for fault-tolerant quantum computing: progress, challenges and outlook*, Nano Futures 7, 032003 (2023); arXiv:2303.00054** [^realtime]. Why ~1 µs/cycle for superconducting qubits is a hard wall, what the backlog exponential looks like, and the FPGA/ASIC landscape. This is the framing document for §3.2.

### 1.5 ML decoding specifically

- **Lee, Hur, Jae, Park, *Machine Learning Approaches to Decoding Topological Quantum Codes*, arXiv:2608.15760 (Aug 2026)** [^mlsurvey]. A 42-page survey (to appear in Springer's *Quantum Error Decoding*) covering decoding-as-a-learning-problem, and discriminative / generative / reinforcement-learning approaches, with an explicit focus on the expressivity–scalability–latency trade-off and on real-time constraints. **Use this as the map of the design space** — see §3.3.
- Earlier landmarks the survey builds on, useful for reading order: Baireuther et al., NJP 21, 013003 (2019) (recurrent decoder, circuit-level noise) [^baireuther]; Varbanov et al. (recurrent decoder on Sycamore data) [^varbanov]; Lange et al., arXiv:2307.01241 (graph neural networks) [^gnn].

**Suggested reading order:** Roffe → Fowler §§I–V → Stim docs + `DEM.md` → deMarti iOlius survey → Battistel real-time review → Lee et al. ML survey → the three papers in §2.

---

## 2. Three papers where ML improves decoding

The framing that matters for us: **each of these is a case where a published algorithm meets publicly available data.** You do not need a quantum computer to reproduce the core claims — you need Stim, the public experimental datasets, and a GPU/TPU. What is *not* released in any of the three is the trained model itself; the reproduction path is re-implementing the architecture from Methods and retraining.

### 2.1 AlphaQubit — learning to decode from data
**Bausch et al., *Learning high-accuracy error decoding for quantum processors*, Nature 635, 834 (2024)** [^alphaqubit]

**What it is.** A recurrent, transformer-based neural decoder for the surface code. Each stabilizer carries its own state vector; one error-correction round updates all of them via a "syndrome transformer" stack (attention + convolutions); a readout network maps the final state to the probability that the logical observable flipped. It is trained by minimizing cross-entropy against the true logical label — i.e. it is a plain binary classifier over syndrome histories.

**The two-stage training recipe** (this is the transferable idea):
1. **Pretrain** on cheap synthetic data — up to ~2×10⁹ samples from either a device-agnostic circuit-depolarizing model (SI1000) or a DEM fitted to hardware ($p_{ij}$ correlations or XEB-derived Pauli weights). *This is exactly the Path A / Path B distinction in* `DEM_prior.md`.
2. **Finetune** on a limited budget of real experimental samples (they partition 325,000 Sycamore samples into train/validation, holding out a fold for test).

**Results.**
- Sycamore memory experiment: logical error per round **2.901% ± 0.023% at d=3** and **2.748% ± 0.015% at d=5** (ensemble of 20 models), vs. 3.028%/2.915% for the tensor-network decoder, 3.117%/3.059% for MWPM-BP, and 4.015%/4.356% for PyMatching. Error-suppression ratio Λ = 1.056 ± 0.010.
- On simulated data with a richer noise model (Pauli+: leakage, cross-talk, soft I/Q readout) it keeps the lead **up to distance 11**, beating even soft-input-augmented correlated matching.
- **Generalization:** trained on ≤25 rounds, it holds accuracy out to **10⁵ rounds**.
- **Calibrated confidence:** the output probability is well calibrated, so it doubles as a post-selection signal — rejecting 0.2% of shots cuts the error rate ~20×; 1% rejection gives ~10×; 10% gives ~790×. Directly useful for magic-state distillation.

**Reproducibility handle.** The Sycamore d=3/d=5 dataset — circuits, samples, and several decoders' predicted corrections — is public: **Zenodo 10.5281/zenodo.6804040** (315 MB) [^zenodo_sycamore]. Pretraining data is regenerable from Stim + SI1000. The architecture is fully specified in the Methods. As of this writing I could not find an official code/weights release, so reproduction means re-implementation — which is the point of the exercise anyway.

**Honest limitation the authors flag:** throughput. AlphaQubit is far slower than the 1 µs/round target. Which leads directly to:

### 2.2 AlphaQubit 2 — the same idea, made scalable and real-time
**Senior, Edlich, Heras, et al., *A scalable and real-time neural decoder for topological quantum codes*, arXiv:2512.07737v2 (Mar 2026)** [^alphaqubit2]

**What changed architecturally** (this is the interesting part for §3.3):
- **Interleaved, not nested.** AQ1 put transformers *inside* a heavy recurrent core. AQ2 uses **lightweight recurrent layers interleaved with transformer layers** (`RNN; RNN; 3×transformer; RNN; 3×transformer; RNN; 3×transformer; RNN`). Because the transformer ops no longer sit inside the recurrence, they can be evaluated **in parallel across many cycles** at inference — the single biggest throughput win.
- **Temporal compression.** Groups of 3–6 consecutive measurement cycles are concatenated and projected into one embedding before processing — "no loss in accuracy," and the network then runs at a correspondingly lower rate. *Note this: it is learned compression of the syndrome stream, and it rhymes with the hardware compression in §3.2.*
- **No convolutions**; mean-pooling + **cross-attention** readout; **spatial RoPE** over physical qubit coordinates, plus normalized (x, y) position embeddings so one model covers all code distances.
- **AQ2-RT** (real-time variant): fewer layers (1 RNN, 2–3 transformer, 1 RNN), 128-channel per-stabilizer state (256 for Willow), and a **Griffin-style element-wise gated recurrence (RG-LRU)** instead of the projection-based RNN.
- Training tricks: curriculum learning (small d → large d), an auxiliary loss predicting a noise-free idealized readout, random input dropout, and ensembling at the largest distances.

**Results.**
- **Accuracy at scale:** 7.3 × 10⁻¹¹ logical error per cycle on the **distance-23 surface code** (120-cycle, SI1000 at 0.15%); 8.0 × 10⁻¹¹ on the **distance-27 Bell-flagged colour code**. Close to the near-optimal-but-slow Libra; far better than PyMatching. For the colour code it is *orders of magnitude faster* than any comparably accurate decoder.
- **Real time:** AQ2-RT sustains **< 1 µs/cycle** for the surface code to **d = 11** (241 qubits) and the superdense colour code to **d = 9** (181 qubits), **on commercial ML accelerators (Trillium TPUs) — no ASIC or FPGA required.** This is the first ML decoder to hit the superconducting real-time budget at these scales. Cost: a modest accuracy trade (3.73 × 10⁻⁶ vs 1.89 × 10⁻⁶ at d=11).
- **Temporal stability:** trained on ≤168 cycles, evaluated to **10⁶ cycles** with stable LER; robust across a **16× range of noise strength** without retraining.
- **Experimental validation:** on the 105-qubit **Willow** data (d = 3, 5, 7), AQ2 matches AQ1 and beats Libra; AQ2-RT beats the real-time matching decoder used in the original Willow paper.

**Reproducibility handle.** The Willow dataset is public: **Zenodo 10.5281/zenodo.13273331** (112.5 GB — `google_105Q_surface_code_d3_d5_d7.zip` is only 5.7 GB and is the one you want) [^zenodo_willow]. Simulated data is Stim + SI1000 at p = 0.15%. The architecture is specified layer-by-layer in Methods A.1, including the exact layer sequences quoted above. A faithful small-scale re-implementation at d = 3–5 is a realistic single-person project.

### 2.3 Reinforcement learning control of QEC — using the *same* data stream to fix the device
**Sivak, Morvan, et al. (Google Quantum AI & DeepMind), *Reinforcement learning control of quantum error correction*, Nature 655, 879 (2026)** [^rlqec]

**Why it belongs in a decoder note.** It is not a decoder — it is the other thing you can do with a detection-event stream. The insight: **error-detection events are not only a decoding input, they are a free, continuously available learning signal about the analogue control parameters of the device.** Instead of halting computation to recalibrate, an RL agent perturbs >1,000 control parameters during the QEC cycle and reads out the change in detection-event statistics.

**The key trick — a differentiable surrogate.** Logical error rate is a terrible optimization objective (exponentially expensive to resolve, and unavailable in general during a real computation). They instead optimize $C = \mathbb{E}[D]$, the mean rate of error-detection events. Using $\varepsilon_L \propto \Lambda^{-d/2}$ they show $\nabla \log \varepsilon_L \approx \frac{d+1}{2} \nabla \log C$ — so minimizing the detection rate minimizes the logical error rate, at a cost that is *independent of code distance*. Sparsity from the **detecting-region** structure (each detector depends only on gates in its own space-time region; ~302 parameters per detector) is exploited via a **factor-graph** representation with gradient masking.

**Results.**
- **20% additional LER suppression** on top of an already exhaustively expert-calibrated Willow processor, consistently across runs.
- Against injected drift: **3.5× improvement in LER stability**; under natural drift, **24% mean LER reduction and 2.4× stability improvement**, rising to **31% / 3.5×** when decoder parameters are steered too.
- **Record QEC performance:** distance-7 surface code at $\varepsilon_L = 7.72(9) \times 10^{-4}$ (decoded with **AlphaQubit 2**), distance-5 colour code at $8.19(14) \times 10^{-3}$ (Tesseract).
- Simulations scale to a **distance-15** surface code with ~40,000 control parameters; convergence rate is **independent of system size**.
- Identifies a critical drift frequency (~1/150 epochs) below which real-time steering wins.

**Reproducibility handle.** Data at **Zenodo 10.5281/zenodo.17566521**; **the code is explicitly proprietary and not released.** So this one is reproducible at the level of *analysis and simulation* (the surrogate-objective argument and the factor-graph structure are fully described, and a policy-gradient agent on a Stim-simulated device with drifting parameters is a legitimate self-contained project), not at the level of rerunning their hardware loop.

### Comparison

| | AlphaQubit (2024) | AlphaQubit 2 (2026) | RL control (2026) |
|---|---|---|---|
| ML replaces | the decoder | the decoder | the calibration loop |
| Model | recurrent + syndrome transformer | interleaved light RNN + transformer, temporal compression | Gaussian policy over control params, policy gradient |
| Best result | 2.75% LER/round, d=5 Sycamore; d≤11 simulated | 7.3×10⁻¹¹ LER/cycle at d=23; <1 µs/cycle to d=11 | 20% LER suppression; 3.5× drift stability; d=7 LER 7.7×10⁻⁴ |
| Speed | far from real time | **real time on stock TPUs** | n/a (outer loop) |
| Public data | Zenodo 6804040 (Sycamore) | Zenodo 13273331 (Willow) | Zenodo 17566521 |
| Public code | no | no | no (stated proprietary) |
| Reproduce by | re-implement + Stim pretrain + finetune | re-implement + Stim SI1000 | simulate drifting device in Stim + policy gradient |

---

## 3. Understanding the data structure → things to build

### 3.1 What the data actually is

Strip away the physics and the decoder input is one object:

```
detection_events : bool[n_shots, n_rounds, n_detectors]     # sparse: ~0.5–6% of entries are 1
final_measurements : bool[n_shots, n_data_qubits]           # last round, data-qubit readout
observable_flips : bool[n_shots, n_observables]             # the label
```

with two optional enrichments that the ML decoders exploit and matching decoders mostly cannot:

- **soft inputs** — the analogue I/Q readout value instead of a thresholded bit;
- **leakage flags** — whether a qubit left the computational subspace.

Structural facts worth internalizing before designing anything:

- **It is extremely sparse.** Detection event density is ~5% at SI1000 p = 0.15%, ~6% on Sycamore hardware. Most of the tensor is zeros. This is what both §3.2 and §3.3 exploit, in hardware and in software respectively.
- **It is causal and streaming.** Round *n* never depends on round *n+1*. Any decoder with a fixed per-cycle cost and constant memory (AQ2's design point) can run forever.
- **It is geometrically structured.** Each detector has (x, y, t) coordinates. AQ2 encodes this with RoPE over spatial coordinates; matching decoders encode it as graph adjacency; a compressor can exploit it as spatial locality of active syndromes.
- **The formats are standard.** Stim's `.b8` / `.dets` / `.01` sampler outputs and `.dem` detector-error-model files. Everything in `DEM.md` is the statistical model *of this tensor*.

Concretely: `stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=10, ...)` then `.compile_detector_sampler().sample(shots, separate_observables=True)` gives you the exact object above in about four lines. Start there before touching any public dataset.

### 3.2 Project A — compress the syndrome stream (an FPGA project)
**Tao, Knapen, Mack, Ravi, Zhang, Saligane, Sylvester, *CryoZip: An Efficient Cryogenic Compressor for Quantum Error Correction Syndromes*, arXiv:2606.30805 (2026)** [^cryozip]

**The problem.** At scale, the wire from the cryostat to the room-temperature decoder is a bandwidth and power bottleneck — every syndrome bit from every stabilizer in every round has to get out of the fridge. Cooling power at 4 K is scarce, so anything you put down there must be tiny.

**The scheme.** Two stages, exploiting exactly the sparsity noted in §3.1:
1. **Sparse Distance (SD) encoding** — instead of transmitting the bit vector, transmit the *gaps* between consecutive active syndromes.
2. **Huffman encoding** of the resulting distance symbols, using their empirical frequency distribution.

Implemented as a pipeline with FIFO buffering and dual lookup tables, placed at **4 K** alongside a lightweight predecoder, processing one round of syndrome measurements at a time within a **100 ns per-round latency budget**.

**Numbers.**

| Metric | Result |
|---|---|
| Surface code compression | up to **48.3×** (1.8× better than prior art) |
| Bivariate bicycle codes | 4.38–42.46× |
| Colour codes | 3.21–30.03× |
| With predecoder integration | > 14,238× total bandwidth reduction |
| Energy (hardware alone) | 4–26× savings |
| Energy (with predecoder + realistic interface overhead) | up to 42× |
| Sliding-window vs non-windowed | 12.2× area, 8.7× power reduction |
| Technology | 22 nm FDSOI CMOS, characterized at 4 K |
| Evaluation | surface code d = 3–21, colour, BB codes; SI1000 at p = 1e-3, 5e-4, 1e-4 |

**Source code: <https://github.com/ReaLLMASIC/CryoZip>** — this is the one paper in this note with an actual code release.

**Suggested FPGA project.** The paper's target is a cryo-CMOS ASIC; an FPGA reimplementation is a tractable stand-in and a genuinely useful artifact:

1. Generate syndrome streams with Stim across d = 3–11 and several noise strengths. Measure the empirical detection-event density and the entropy of the SD-distance distribution — this sets the achievable compression before you write any HDL.
2. Implement SD + Huffman in Verilog/VHDL (or HLS). Fixed Huffman table first, from offline statistics; adaptive later.
3. Verify against the CryoZip reference implementation on identical inputs.
4. Measure **throughput and latency against the 100 ns/round budget**, and resource usage (LUTs, BRAM, power estimate) as a function of code distance. The sliding-window variant is where the area/power win is — implement both and reproduce the 12.2×/8.7× claim on FPGA.
5. Baselines to beat: raw transmission, run-length encoding, generic LZ. Report compression ratio *and* energy per bit.
6. Stretch: close the loop — decompress on the host and feed PyMatching, confirming bit-exact recovery and measuring end-to-end added latency.

The nice property of this project is that every input is free and public (Stim), the reference is open source, and the success criterion is a number, not a judgement.

**Adjacent prior art to read before starting:** *Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder*, arXiv:2605.04892 (2026) [^fpganeural] — the complementary move, putting the *decoder* itself on an FPGA rather than the compressor. Between the two you get the full picture of what belongs at 4 K, what belongs on an FPGA at 300 K, and what belongs on a TPU.

### 3.3 Project B — build a leaner model architecture
**Lee, Hur, Jae, Park, *Machine Learning Approaches to Decoding Topological Quantum Codes*, arXiv:2608.15760 (2026)** [^mlsurvey]

To be precise about what this is: it is a **42-page survey/review chapter**, not a new architecture. Its value here is as the **map of the design space** — it organizes decoding-as-learning into discriminative, generative, and reinforcement-learning families, walks through the neural building blocks underneath contemporary decoders, reviews memory-experiment benchmarks, and — most relevantly — frames everything against the **expressivity vs. scalability vs. latency** trade-off and the real-time decoding constraint.

Use it to choose where to attack. The openings the AQ2 authors themselves name as unexplored or unfinished are the obvious starting set:

- **Distillation.** AQ2 uses ensembles of 2–3 models for peak accuracy at large distances and explicitly says a distilled single "student" model might match the ensemble without the cost. Nobody has done it.
- **Low-precision arithmetic.** Named as an expected route to further throughput. Post-training int8 quantization of AQ2-RT is a well-defined, self-contained experiment.
- **Sparse / local attention.** AQ2's spatial mixing is full self-attention across all $d^2-1$ stabilizers, which is $O(d^4)$ per layer per time step. The physics is local. Restricting attention to a spatial neighbourhood (plus a few global tokens) is the obvious architectural bet, and directly testable against their published d = 3–11 numbers.
- **More aggressive temporal compression.** They compress 3–6 cycles with no accuracy loss; where is the actual knee?
- **Windowing** — decode overlapping space-time windows independently, which is also what makes logical-operation decoding tractable.
- **Pruning** of the per-stabilizer state (128 channels in AQ2-RT — how far down does it go at d = 3–5?).

Evaluation protocol for any of these: the metric is a **Pareto point on (logical error per cycle, time per cycle)**, exactly as in AQ2's Fig. 4a. Report both, on Stim-generated SI1000 data at fixed p, plus the Willow experimental set if you get that far.

### 3.4 A concrete ladder

A sequence that builds capability without requiring anything you don't already have:

1. **Baseline.** PyMatching on Willow d = 3/5/7 from Zenodo 13273331. Reproduce the published LER/cycle. *(Deliverable: you can read the public data format.)*
2. **Smallest possible neural decoder.** A 2-layer GRU on Stim SI1000 d = 3, 10 rounds. Beat PyMatching or understand why you don't. *(Deliverable: the training loop.)*
3. **DEM-informed pretraining.** Fit a DEM from detector correlations (Path B in `DEM_prior.md`), generate pretraining samples from it, finetune on experimental data. Reproduce AlphaQubit's central claim — that a fitted-DEM prior beats a device-agnostic one. *(Deliverable: the link between this note and `DEM_prior.md` becomes operational.)*
4. **Scale and speed.** Add the AQ2 ingredients one at a time — temporal compression, interleaved RNN/transformer, RoPE — and plot the accuracy/throughput Pareto front after each. *(Deliverable: an architecture ablation nobody has published.)*
5. **Compression front-end.** §3.2, on FPGA. *(Deliverable: hardware.)*

---

## 4. Takeaways

- **The decoder's input is a sparse binary space-time tensor**, and `DEM.md` is the statistical model of that tensor. Every ML decoder in this note is a sequence model over that object; every hardware trick in §3 exploits its sparsity, causality, or locality.
- **ML decoding has moved from "competitive" to "state of the art."** AlphaQubit beat every conventional decoder on real Sycamore data in 2024; AlphaQubit 2 reached 10⁻¹¹ logical error per cycle at distance 23 *and* real-time throughput on stock accelerators in 2026 — the first time accuracy and speed have been met together.
- **The data is public even though the models aren't.** Sycamore (Zenodo 6804040), Willow (Zenodo 13273331), and the RL control data (Zenodo 17566521) are all downloadable; Stim regenerates unlimited training data. The architectures are specified in the Methods. Reproduction is a re-implementation exercise, not a data-access problem — which makes it a good exercise.
- **The same detection-event stream is dual-use.** Sivak/Morvan et al. turn it into a reward signal for continuous calibration, with a surrogate objective whose cost does not grow with code distance. Decoder quality and control quality are coupled — their record d=7 LER used AlphaQubit 2 as the decoder.
- **Two buildable directions.** *Downward into hardware:* CryoZip's SD+Huffman compressor (48× on surface codes, 100 ns/round, open source) is a well-posed FPGA project with free inputs and numeric success criteria. *Sideways into architecture:* distillation, quantization, and sparse attention are named-but-untried openings in AQ2, and arXiv:2608.15760 is the map of that space.

---

## References

[^fowler2012]: A. G. Fowler, M. Mariantoni, J. M. Martinis, A. N. Cleland, *Surface codes: Towards practical large-scale quantum computation*, Phys. Rev. A **86**, 032324 (2012). <https://arxiv.org/abs/1208.0928>.

[^terhal2015]: B. M. Terhal, *Quantum error correction for quantum memories*, Rev. Mod. Phys. **87**, 307 (2015). <https://arxiv.org/abs/1302.3428>.

[^roffe2019]: J. Roffe, *Quantum error correction: an introductory guide*, Contemp. Phys. **60**, 226 (2019). <https://arxiv.org/abs/1907.11157>.

[^stim]: C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum **5**, 497 (2021). <https://arxiv.org/abs/2103.02202>. Code: <https://github.com/quantumlib/Stim>.

[^google2023]: Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*, Nature **614**, 676 (2023).

[^google2025]: Google Quantum AI, *Quantum error correction below the surface code threshold*, Nature **638**, 920 (2025).

[^decodingreview]: A. deMarti iOlius, P. Fuentes, R. Orús, P. M. Crespo, J. Etxezarreta Martinez, *Decoding algorithms for surface codes*, Quantum **8**, 1498 (2024). <https://arxiv.org/abs/2307.14989>.

[^pymatching]: O. Higgott, *PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching*, arXiv:2105.13082 (2021); O. Higgott, C. Gidney, *Sparse blossom: correcting a million errors per core second with minimum-weight matching*, Quantum **9**, 1600 (2025). Code: <https://github.com/oscarhiggott/PyMatching>.

[^tesseract]: L. Aghababaie Beni, O. Higgott, N. Shutty, *Tesseract: A search-based decoder for quantum error correction*, arXiv (Mar 2025).

[^libra]: C. Jones, *Improved accuracy for decoding surface codes with matching synthesis*, arXiv:2408.12135 (2024).

[^chromobius]: C. Gidney, C. Jones, *New circuits and an open source decoder for the color code*, arXiv (Dec 2023).

[^realtime]: F. Battistel, C. Chamberland, K. Johar, R. W. J. Overwater, F. Sebastiano, L. Skoric, Y. Ueno, M. Usman, *Real-time decoding for fault-tolerant quantum computing: progress, challenges and outlook*, Nano Futures **7**, 032003 (2023). arXiv:2303.00054. <https://arxiv.org/abs/2303.00054>.

[^mlsurvey]: C. Lee, T. Hur, J. Jae, D. K. Park, *Machine Learning Approaches to Decoding Topological Quantum Codes*, arXiv:2608.15760 (Aug 2026). To appear in *Quantum Error Decoding*, Springer Quantum Science and Technology series. <https://arxiv.org/abs/2608.15760>.

[^baireuther]: P. Baireuther, M. D. Caio, B. Criger, C. W. J. Beenakker, T. E. O'Brien, *Neural network decoder for topological color codes with circuit level noise*, New J. Phys. **21**, 013003 (2019).

[^varbanov]: B. M. Varbanov, M. Serra-Peralta, D. Byfield, B. M. Terhal, *Neural network decoder for near-term surface-code experiments*, Phys. Rev. Research **7**, 013029 (2025); arXiv:2307.03280 (2023). <https://arxiv.org/abs/2307.03280>.

[^gnn]: M. Lange et al., *Data-driven decoding of quantum error correcting codes using graph neural networks*, arXiv:2307.01241 (2023). <https://arxiv.org/abs/2307.01241>.

[^alphaqubit]: J. Bausch, A. W. Senior, F. J. H. Heras, T. Edlich, A. Davies, M. Newman, C. Jones, K. Satzinger, M. Y. Niu, S. Blackwell, G. Holland, D. Kafri, J. Atalaya, C. Gidney, D. Hassabis, S. Boixo, H. Neven, P. Kohli, *Learning high-accuracy error decoding for quantum processors*, Nature **635**, 834 (2024). DOI: <https://doi.org/10.1038/s41586-024-08148-8>.

[^alphaqubit2]: A. W. Senior, T. Edlich, F. J. H. Heras, L. M. Zhang, O. Higgott, J. S. Spencer, T. Applebaum, S. Blackwell, J. Ledford, A. Žemgulytė, A. Žídek, N. Shutty, A. Cowie, Y. Li, G. Holland, P. Brooks, C. Beattie, M. Newman, A. Davies, C. Jones, S. Boixo, H. Neven, P. Kohli, J. Bausch, *A scalable and real-time neural decoder for topological quantum codes*, arXiv:2512.07737v2 (11 Mar 2026). <https://arxiv.org/abs/2512.07737>.

[^rlqec]: V. Sivak, A. Morvan, M. Broughton, R. G. Cortiñas, et al. (Google Quantum AI & Google DeepMind), *Reinforcement learning control of quantum error correction*, Nature **655**, 879 (2026). DOI: <https://doi.org/10.1038/s41586-026-10759-2>. Data: Zenodo <https://doi.org/10.5281/zenodo.17566521>. Code: proprietary, not released.

[^fpganeural]: *Real-time Surface-Code Error Correction Using an FPGA-based Neural-Network Decoder*, arXiv:2605.04892 (2026). <https://arxiv.org/abs/2605.04892>. (Not yet read in full — listed as adjacent prior art for the §3.2 project.)

[^cryozip]: G. Tao, A. Knapen, J. Mack, G. S. Ravi, Q. Zhang, M. Saligane, D. Sylvester, *CryoZip: An Efficient Cryogenic Compressor for Quantum Error Correction Syndromes*, arXiv:2606.30805 (2026). <https://arxiv.org/abs/2606.30805>. Code: <https://github.com/ReaLLMASIC/CryoZip>.

[^zenodo_sycamore]: Google Quantum AI, *Data for "Suppressing quantum errors by scaling a surface code logical qubit"*, Zenodo (2022). DOI: <https://doi.org/10.5281/zenodo.6804040>. 315 MB; circuits, samples, and several decoders' predicted corrections. CC-BY-4.0.

[^zenodo_willow]: Google Quantum AI, *Data for "Quantum error correction below the surface code threshold"*, Zenodo (2024). DOI: <https://doi.org/10.5281/zenodo.13273331>. 112.5 GB total; `google_105Q_surface_code_d3_d5_d7.zip` (5.7 GB) is the surface-code memory data used by AlphaQubit 2. CC-BY-4.0.

---

*This is a living document. Next steps I plan to take: (a) run step 1 of the ladder in §3.4 — PyMatching on the Willow d=3/5/7 data — and record the LER/cycle numbers here; (b) fill in the ML-survey section once I have read arXiv:2608.15760 in full rather than in summary; (c) cross-link the DEM-fitting step in §3.4.3 back to the Path-B worked example in `DEM_prior.md`.*
