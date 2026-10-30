# Cryogenic hardware constraints on decoder size (2026-10-29 → )

Living entry. It covers what a ≤4 K placement does to the size of the model we can run, and the hardware theory behind that. Status tags: **[src]** comes from a source listed in §8. **[bg]** is Claude's background knowledge and needs a citation before it goes in a deliverable. **[est]** is a back-of-envelope estimate built here.

**Scope note.** `PROJECT.md` says the project is "entirely classical", and the FPGA stretch goal runs at room temperature. In this entry, 4 K is the *design constraint the model is sized against*. On a room-temperature board we can measure resources and power and compare them with a 4 K budget; we cannot test 4 K operation itself. Confirm this framing with Dr. Kim (§7).

---

## 1. Thermodynamics: heat is the budget, not temperature

A cryostat stage holds its temperature only while the heat load stays below that stage's cooling power. At steady state the stage sits at the temperature T where

  P_load = Q_cool(T).

Q_cool(T) is the cooler's capacity curve, and it falls steeply as T drops. Extra computation does not "raise the temperature" smoothly. It adds load P_load, and the stage stays put until the load passes Q_cool. After that, T climbs along the curve until the load and the capacity balance again, or the qubits stop working.

| stage | typical cooling power | source |
|---|---|---|
| 4 K (pulse tube, PT2) | 1–2 W conservative; ~5 W per pulse tube on large systems (Bluefors XLD1000) | [src] Oxford Instruments; arXiv:2604.16216 |
| still (~700 mK) | 10–20 mW | [src] |
| cold plate (~100 mK) | 200–500 µW | [src] |
| mixing chamber (~20 mK) | 10–25 µW | [src] |

This 4 K budget is shared. Control and readout electronics, the heat that wiring conducts in, and attenuators all draw on it before any decoder does. The rule of thumb is to intercept heat at the warmest stage that can take it [src]. Below 4 K the budget is µW to mW, so CMOS is ruled out and only superconducting logic fits (§5).

**Wall-plug cost.** Removing heat Q at T_c and rejecting it at T_h takes at least

  W / Q ≥ (T_h − T_c) / T_c = 296 / 4 ≈ 74   (Carnot, 4 K).

Real coolers need about 10³–10⁴ W of wall power per W removed at 4 K [bg]. For example, a pulse tube that gives ~1.5 W at 4 K runs on a ~10 kW compressor. So a 1 W decoder at 4 K costs ~kW at the wall, while the same decoder at 300 K costs 1 W. That trade only pays when putting the decoder at 4 K buys something else: lower latency, or less data on the wires leaving the fridge (CryoZip's argument, `project_docs_context/decoder_ML.md` §3.2).

## 2. Power of a digital computation

CMOS power is [bg]

  P = P_static + α · C · V² · f

where α is the activity factor, C the switched capacitance, V the supply voltage and f the clock. The dynamic term is energy per operation × operations per second. So for a decoder that runs at the round rate f_round:

  P_dyn ≈ f_round · Σ_ops (n_op · e_op)

**Energy per operation**, 45 nm ASIC (Horowitz, ISSCC 2014) [bg]:

| op | energy |
|---|---|
| 8-bit int add | 0.03 pJ |
| 8-bit int mult | 0.2 pJ |
| 32-bit float mult | 3.7 pJ |
| 32-bit read, 8 KB SRAM | 5 pJ |
| 32-bit read, DRAM | ~640 pJ |

What follows from the table:
- **The unit for decoders is the MAC, not the FLOP.** Hardware inference runs in fixed point. "FLOPs" in ML papers usually means 2 × MACs.
- **Moving data costs more than arithmetic.** One SRAM read costs ~25 INT8 multiplies. Weights have to sit in registers or BRAM right next to the multipliers.
- **Shorter words help more than linearly.** Multiplier energy and area grow roughly with bits², which is why INT4 matters (Yan et al. 2605.12046, `MWPM.md` §7.4).
- **An FPGA is less efficient than an ASIC on the same node.** It takes roughly 35× the area, 3–4× the delay and ~14× the dynamic power (Kuon & Rose 2007) [bg], because of programmable routing and configuration overhead.

**CMOS at 4 K** [bg]:
- Subthreshold swing scales with kT/q. It would fall from 60 mV/decade toward ~1 mV/dec at 4 K, but band-tail states make it saturate around 10–20 mV/dec. Leakage drops by orders of magnitude, so **P_static nearly vanishes**.
- V_th rises by ~0.1–0.2 V and mobility goes up.
- The dynamic CV²f term barely changes, so **dynamic power is the budget**.
- In bulk CMOS, freeze-out and kink effects complicate device models. FDSOI behaves better, which is why CryoZip used 22 nm FDSOI [src].

**FPGAs at 4 K.**
- A Xilinx Artix-7 is fully functional at 4 K; an Intel Cyclone V fails below ~30 K (Homulle & Charbon) [src].
- Spartan-3 and Spartan-6 also work at 4 K [src].
- Most voltage regulators and decoupling capacitors misbehave at 4 K. Their fix was firmware that holds power draw constant so the internal supply stays stable [src].
- Usable 4 K FPGA hardware therefore exists, but only on old 28 nm-class parts with board-level workarounds.

## 3. Three constraints that bound model size

A decoder sized for 4 K has to satisfy all three at once. Let N_MAC be the multiply-accumulates per QEC round and N_par the parameter count.

**(a) Energy per round.**

  N_MAC · e_MAC,eff ≤ P_budget / f_round

Here e_MAC,eff includes the weight and activation traffic. f_round ≈ 1 MHz (the 1 µs cycle).

| scenario | P_budget per logical qubit | energy per round | N_MAC @ 1 pJ (cryo ASIC-ish) | N_MAC @ 10 pJ (FPGA-ish) |
|---|---|---|---|---|
| single-logical demo (us) | 100 mW | 100 nJ | ~10⁵ | ~10⁴ |
| scaled machine, ~1000 logicals sharing ~1–5 W | ~1 mW | 1 nJ | ~10³ | ~10² |

[est] The e_MAC values are order-of-magnitude placeholders. Replace them with Vivado power-report numbers once we have a synthesized design.

**(b) Latency and throughput per round.** At clock f_clk, one round leaves C = f_clk / f_round cycles. That is ~200 cycles at 200 MHz.
- Throughput: N_MAC ≤ N_mult · C, where N_mult is the number of physical multipliers (DSP slices, plus LUT multipliers at INT4).
- Latency: the pipeline depth through all layers must fit within the reaction-time budget. Whether that budget is per round or after the last round is still an open question (README).
- In hls4ml terms, the *reuse factor* R sets how many MACs share each multiplier. N_mult = N_par / R, and each layer's latency is about R cycles. R = 1 means fully unrolled: lowest latency, one multiplier per weight.

Example: an Artix-7 100T has 240 DSP48 slices [bg]. So N_MAC ≤ 240 × 200 ≈ 5×10⁴ per round with DSPs alone. INT4 packing or LUT multipliers can add a few times that.

**(c) On-chip memory.** There is no DRAM at 4 K: DRAM is too hot and too far, and its access costs 640 pJ. So all weights live in BRAM or registers:

  N_par · bits ≤ on-chip bits

The Artix-7 100T has ~4.9 Mb of BRAM [bg], so it holds ~1.2M INT4 weights. **This constraint rarely binds. (a) and (b) bind first.**

**What happens to (a) and (b) together.** Both land at **N_MAC ~ 10⁴–10⁵ per round** for a single-logical-qubit FPGA demo, and at **~10³** for a scaled 4 K deployment.

### 3.1 Converting an architecture to MACs per round (d=3: 8 stabilizers, 8 detectors per round)

Formulas [bg], with N tokens of width D, hidden size H and input size I:

| block | MACs per round |
|---|---|
| dense layer, I → O | I·O |
| transformer layer, N tokens × D | ≈ 12·N·D² + 2·N²·D (QKV and output projections 4ND², FFN with 4× expansion 8ND², attention 2N²D) |
| LSTM / GRU cell | 4·H·(H+I) / 3·H·(H+I) |
| conv, k×k, C_in → C_out on an h×w grid | h·w·k²·C_in·C_out |

Transformer at N = 8 (one token per stabilizer, AlphaQubit-style), per layer [est]:
- D = 128: ~1.6×10⁶ MACs. Three layers is ~5×10⁶, **about 100× over** budget (a).
- D = 32: ~10⁵ MACs per layer. That fits the demo budget at INT4 with 1–2 layers.

**So the 4 K constraint puts the model in the D ≈ 16–32 regime, 1–2 layers or a small recurrent cell, at INT4.** That is the regime of Overwater et al. 2022: on an Artix-7, only their d=3 feed-forward designs fit [src].

**Compare against a lookup table.** At d=3 a lookup-table decoder is a real competitor (LILLIPUT, Das et al. ASPLOS 2022) [bg]. A neural model at this size has to beat a LUT built on the same budget, not just MWPM.

## 4. Why 4 K at all

Room temperature is the standard home for decoders today: CPU/GPU/TPU (AQ2-RT runs on TPUs), FPGA (Riverlane), or ASIC.

Three things argue for moving decoding down to 4 K:
1. **I/O bandwidth.** Every syndrome bit from every round has to leave the fridge, and the wires carry heat in.
2. **Latency.** Reaction time includes the cable round trip.
3. **Scale.** The number of wires grows with the qubit count.

The usual answer is a **split hierarchy** [src, `decoder_ML.md` §3.2]:
- a tiny *predecoder* or *compressor* at 4 K (CryoZip: 100 ns per round, 22 nm FDSOI);
- the heavy decoder at 300 K.

Our model is most plausible at 4 K as that **predecoder**. It would resolve easy, local syndromes and pass only the hard residue up, which is the role Ueno's NEO-QEC pre-processor plays.

## 5. What is typical for a superconducting architecture

| technology | where | energy scale | capacity | decoder examples |
|---|---|---|---|---|
| room-temperature CPU/GPU/TPU/FPGA | 300 K | pJ–nJ/op, unconstrained | large | PyMatching, AQ2-RT on TPU, Riverlane FPGA/ASIC |
| cryo-CMOS ASIC | 4 K | ~pJ/op, static power ≈ 0 | moderate; fixed function | CryoZip compressor [src]; AFS union-find (Das et al. HPCA 2022) [bg]; Intel Horse Ridge (control, not decoding) [bg] |
| cryo-FPGA | 4 K | ~10× cryo-ASIC | Artix-7 class | qubit control/readout demos (Homulle/Charbon) [src]; no decoder demo found yet |
| SFQ / ERSFQ / AQFP superconducting logic | 4 K and below | ~I_c·Φ₀ ≈ 10⁻¹⁹ J per switch [bg]; static bias power dominates in plain RSFQ | 10⁴–10⁶ junctions, no dense memory | QECOOL/QULATIS (online greedy matching), **NEO-QEC: binarized-NN preprocessor in SFQ, +53% threshold over QECOOL, d=5–13** [src] |

The pattern: **the colder the placement, the smaller and lower-precision the model**. At 4 K in CMOS that means INT4 and ~10³–10⁵ MACs. In SFQ at 4 K and below it means binarized networks.

## 6. What we measure on our room-temperature FPGA

For every candidate model, record:
- N_par, bit width, and N_MAC per round, from the formulas in §3.1;
- after synthesis: LUT / DSP / BRAM usage, latency in cycles, the initiation interval (cycles between rounds), and Vivado's dynamic-power estimate;
- energy per round = P_dyn / f_round, plotted against the §3 budget lines (100 nJ for the demo, 1 nJ at scale);
- the accuracy axis: ε per round against MWPM and against optimal (`mwpm_headroom.md`).

Report static power separately. It overstates the 4 K case, because leakage collapses at 4 K (§2).

## 7. Open questions

- [ ] Is 4 K a real requirement from Dr. Kim, or our own long-term framing? `PROJECT.md` says the work is entirely classical. The answer decides between a cryo-power axis on the plots and a 4 K "budget line" in the discussion section.
- [ ] Get a per-logical-qubit power budget from the literature. Battistel et al. 2303.00054 is the first place to look; the 100 mW and 1 mW figures above are placeholders.
- [ ] Get measured numbers for Artix-7 at 4 K: power at 4 K versus 300 K, and any change in maximum clock. Sources: arXiv 1602.05786, 1709.04190, RSI 10.1063/1.5004484.
- [ ] Read Overwater et al. 2202.05741 for their measured Artix-7 LUT/DSP/latency numbers for d=3 feed-forward decoders. Those are our nearest reference point.
- [ ] Read NEO-QEC 2208.05758 for the BNN size and per-round energy in SFQ.
- [ ] Count AQ2-RT's MACs per round from Methods: stabilizer embedding width and layer count. That shows how far the reproduction target sits from the 4 K regime.
- [ ] Find a citation for the [bg] items: Horowitz energies, Kuon & Rose, cryo-CMOS device physics, LILLIPUT, AFS.

## 8. Sources

- Homulle & Charbon, cryogenic FPGA characterization: [arXiv 1602.05786](https://arxiv.org/pdf/1602.05786v1), [arXiv 1709.04190](https://arxiv.org/pdf/1709.04190), [RSI 10.1063/1.5004484](https://pure.tudelft.nl/ws/portalfiles/portal/53015236/1.5004484.pdf), [EPFL record](https://infoscience.epfl.ch/handle/20.500.14299/147718)
- Overwater, Babaie, Sebastiano, NN decoder hardware cost–performance, IEEE TQE 2022: [arXiv 2202.05741](https://arxiv.org/pdf/2202.05741)
- Ueno et al., NEO-QEC: [arXiv 2208.05758](https://arxiv.org/pdf/2208.05758); QECOOL: [arXiv 2103.14209](https://arxiv.org/abs/2103.14209v1)
- Battistel et al., real-time decoding review: [arXiv 2303.00054](https://arxiv.org/pdf/2303.00054)
- Yan, Li, Du, INT4 on FPGA: [arXiv 2605.12046](https://arxiv.org/pdf/2605.12046)
- Cooling power by stage: [Oxford Instruments, 4 K cooling](https://nanoscience.oxinst.com/resources/blog/enabling-quantum-scale-up-why-high-4-k-cooling-powers-are-standard-on-proteox-dilution-refrigerators), [arXiv 2604.16216](https://arxiv.org/pdf/2604.16216), [RF Essentials, MXC budget](https://rfessentials.com/rf-knowledge-base/how-do-i-calculate-the-thermal-budget-for-microwave-components-at-the-mixing-cha/)
- CryoZip: `project_docs_context/decoder_ML.md` §3.2
