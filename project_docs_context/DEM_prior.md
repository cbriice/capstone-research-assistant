# Detector Error Model (DEM) Priors from Syndrome Data — Three-Paper Comparison

A comparison of three works on building/estimating detector error models (DEMs) from
syndrome data, and what each contributes to the question: *where should a decoder's DEM
prior come from — a device noise model, or the syndrome measurements themselves?*

## The decoding chain (shared context)

```
experiment → raw measurement outcomes (m_a)
           → detectors (fixed parities, deterministic in noiseless circuit)
           → detector events (v_i = 1, deviation from noiseless reference)
           → decode with MWPM, using a DEM as prior
           → predicted logical flip → compare to truth → logical error probability (LEP)
```

A DEM maps each elementary error event `e` to `(p_e, h_e, ℓ_e)`:
its probability `p_e`, the detectors it flips `h_e`, and the logical observables it flips `ℓ_e`.
The **decoder prior** is essentially the set of probabilities `{p_e}` (together with the
support `{(h_e, ℓ_e)}` and the hyperedge decomposition).

- **Conventional prior:** build a circuit-level noise model → assign physical fault rates
  `r_f` → propagate through the circuit → combine same-signature faults by the parity rule
  `p_e = [1 − ∏(1 − 2 r_f)]/2`. Requires device benchmarking/calibration. (RL optimization
  is the other conventional route: tune `p_e` against logical outcomes.)
- **Syndrome-estimated prior:** keep the support fixed, but infer `p_e` directly from
  empirical detector statistics (moments/correlations) of the experimental syndromes —
  no calibration circuits, no microscopic model, no supervised fitting to logical outcomes.

## The three papers

| | **Blume-Kohout & Young** | **Remm et al.** | **Takou … Lidar, Brown** |
|---|---|---|---|
| Ref | arXiv:2504.14643 (Apr 2025) | PRX Research 8, 013044 (2026); recv. Mar 2025 | arXiv:2606.11496 (Jun 2026) |
| Group | Sandia (QPL) | ETH Zürich (Wallraff) + Jülich/Sherbrooke | Duke / USC |
| Type | Theory / framework | Experiment + analytical formula | Experiment |
| Device | None (repetition-code illustration) | One d=3 superconducting surface code | Google Willow + IBM `ibm_miami`; d = 3,5,7 |
| Core object | DEM as the uniquely identifiable noise object | Syndrome correlations → error-channel probs → MWPM weights | Syndrome-estimated DEM as decoder prior |
| Headline | Identifiability + general Walsh–Hadamard formalism w/ derivations | Characterize a broad family of error channels; in-situ decoder calibration | Systematic logical-error reduction across devices |
| Logical-perf. claim | None (estimation only) | Modest; Y-error refinement gave *no significant* gain at d=3 | ~5–10% LEP reduction (up to ~37% IBM r=1), competitive with RL |

### 1. Blume-Kohout & Young — *Estimating detector error models from syndrome data*
**Theory / learnability.** Establishes that Pauli channels cannot be uniquely estimated from
syndrome data (degeneracy), whereas DEMs — "Pauli channels mod stabilizers" — are *exactly*
what is uniquely learnable. Provides a complete formalism via the Walsh–Hadamard transform,
with a change of variables (detector moments → *polarizations*; DEM probabilities →
*attenuations*, which aggregate linearly). Clarifies that the Spitz/Google `p_ij` correlators
are *aggregated probabilities of large classes of DEM events* (reduced DEMs). Supplies the
derivations that earlier empirical works used without proof. No hardware, no decoding benchmark.

### 2. Remm et al. — *Experimentally informed decoding based on syndrome correlations*
**Experimental error-model characterization.** Uses an analytical correlation formula to
extract probabilities for error channels that flip an *arbitrary number* of syndrome elements —
standard X/Z, plus Pauli Y (via correlating X- and Z-type syndromes), hook errors, multiqubit
errors, and correlated **leakage** events spanning up to eight cycles. Uses this to set
MWPM weights in-situ, without a theoretical error model. Distinctive value is the *breadth and
structure* of noise it can diagnose, including collective/correlated and non-Pauli mechanisms.
At d=3 the richer Y-channel knowledge did not significantly lower the logical error, suggesting
LEP there is not Y-limited.

### 3. Takou, Benito, Vezvaee, Lidar, Brown — *Logical error estimation from syndrome data*
**Experimental decoding-strategy benchmark.** A controlled comparison of *prior sources*:
device circuit-level model (SI1000 / IBM-like) vs. RL-optimized vs. estimated-directly-from-
syndromes. Shows the syndrome-direct DEM reduces LEP by ~5–10% relative to device-informed
baselines, competitive with RL — without calibration circuits, benchmarking, decoder
fine-tuning, or supervised fitting — across many subsystems, three distances, both bases, cycle
counts, and two devices. Deliberately separates "rate estimation" from "choice of support."

## How they relate

- Young & Blume-Kohout explicitly note that **Remm** generalized the Spitz correlation
  formulae "without a derivation," and supply that derivation — so Young is the rigorous
  theory underneath Remm's empirical formula.
- **Brown** builds on this estimation line (citing the theoretical works) and pushes it to a
  systematic logical-performance benchmark on two devices.
- Chronological/logical flow:
  **Remm** (what noise is actually there, incl. leakage — characterization on hardware)
  → **Young** (what is provably, uniquely learnable — theory)
  → **Brown** (how to turn it into the best decoder prior in practice — benchmarked).

## Practical reading (who should use which)

- **Understanding device noise, especially collective / correlated / leakage mechanisms →
  Remm.** Diagnostic breadth: resolves Y, hook, multiqubit, and multi-cycle leakage signatures.
- **Choosing a decoding strategy — where the DEM prior should come from, at what cost →
  Brown.** Shows syndrome-direct estimation can replace a benchmarked device model or an
  RL-trained prior with comparable/better LEP and far less overhead.
- **Knowing what is learnable in principle and what the correlators mean → Young &
  Blume-Kohout.** The guarantees and limits behind both empirical methods.

**One-line takeaways**
- Young = *what is learnable in principle* (theory).
- Remm = *what noise is actually there, including collective/leakage* (characterization).
- Brown = *how to turn it into the best decoder prior in practice* (decoding strategy).

---

# How a DEM prior `(p_e, h_e, ℓ_e)` is obtained — worked example

Each DEM event `e` carries three pieces:

- `p_e` — the probability the event fires (a Bernoulli rate),
- `h_e` — the **detector-flip vector** (which detectors light up), an `M`-bit 0/1 vector,
- `ℓ_e` — the **logical-flip vector** (which logical observables it flips), usually 0 or 1.

The pair `(h_e, ℓ_e)` is the event's **detector–logical signature** (its "support"). The two
routes differ in *how much* of the triple they compute from a model versus from data.

| Piece | Conventional (device-model) path | Syndrome-measurement path |
|---|---|---|
| `h_e`, `ℓ_e` | Computed by **propagating each fault** through the noiseless circuit | **Inherited** from a reference DEM's support (Brown); or re-derived from correlations (Remm/Young) |
| `p_e` | Computed from **physical fault rates** `r_f` + parity rule | **Estimated** from empirical detector moments/correlations |
| Needs | A calibrated circuit-level noise model (benchmarking) | Only the syndrome shots already collected |

We use one concrete event throughout: **a data-qubit error between rounds that flips the two
neighbouring stabilizer detectors `i` and `j`, and does not cross the logical observable.**
So its signature is fixed:

```
h_e = (…, 1_i, …, 1_j, …)      ℓ_e = 0        (a "bulk" graphlike edge i–j)
```

## Path A — Conventional (device noise model → Stim → parity rule)

**Step 1 — assign physical fault rates.** A circuit-level noise model (e.g. SI1000 for Google,
or the IBM-like Pauli model) puts a probability `r_f` on every elementary fault location `f`
(gate errors, idle/dephasing, readout flips, reset errors).

**Step 2 — propagate to get `h_e, ℓ_e`.** Each fault `f` is pushed through the noiseless
Clifford circuit (Stim) to see which detectors and logicals it flips. Several distinct faults
can land on the **same** signature `(h_e, ℓ_e)`. Here, suppose two faults map onto our edge `i–j`:

| fault `f` | meaning | rate `r_f` |
|---|---|---|
| `f₁` | idle/dephasing on the data qubit between rounds | `0.0010` |
| `f₂` | depolarizing tail of the CZ gate touching that qubit | `0.0020` |

**Step 3 — combine same-signature faults (parity rule).**

```
p_e = [ 1 − Πf (1 − 2 r_f) ] / 2
    = [ 1 − (1 − 2·0.0010)(1 − 2·0.0020) ] / 2
    = [ 1 − (0.9980)(0.9960) ] / 2
    = [ 1 − 0.994008 ] / 2
    = 0.002996  ≈ 0.0030
```

**Result:** `e ↦ (p_e = 0.0030, h_e = {i, j}, ℓ_e = 0)`. Everything came from the *model*; the
quality depends entirely on how accurately `r_f` was benchmarked.

## Path B — Syndrome-measurement (empirical detector statistics)

Here `(h_e, ℓ_e) = ({i,j}, 0)` is **taken as given** (from a reference DEM's support); only `p_e`
is learned from the data. Let `x_i ∈ {0,1}` be detector `i`'s value (1 = fired). From the
50,000 shots we compute three empirical numbers:

```
⟨x_i⟩     = 0.085      (detector i firing rate)
⟨x_j⟩     = 0.090      (detector j firing rate)
⟨x_i x_j⟩ = 0.0100     (coincidence rate)
```

**Two-point (Spitz) estimator** for the edge probability:

```
Cov_ij = ⟨x_i x_j⟩ − ⟨x_i⟩⟨x_j⟩ = 0.0100 − (0.085)(0.090) = 0.00235

p_ij = 1/2 − 1/2 · sqrt( 1 − 4·Cov_ij / (1 − 2⟨x_i⟩ − 2⟨x_j⟩ + 4⟨x_i x_j⟩) )

  denom = 1 − 0.170 − 0.180 + 0.040 = 0.690
  ratio = 4·0.00235 / 0.690 = 0.013623
  p_ij  = 0.5 − 0.5·sqrt(1 − 0.013623)
        = 0.5 − 0.5·(0.993165)
        = 0.003417  ≈ 0.0034
```

**Result:** `e ↦ (p_e = 0.0034, h_e = {i, j}, ℓ_e = 0)` — same signature, probability now read
straight from the data, **no noise model and no benchmarking used.**

> **Why correlations work (Young's view).** Define the polarization `⟨z_i⟩ = ⟨(−1)^{x_i}⟩`. For
> independent events, `⟨z_i⟩ = Π_{e flips i}(1 − 2p_e)` and `⟨z_i z_j⟩ = Π_{e flips exactly one
> of i,j}(1 − 2p_e)`. Taking `−ln` turns these products into **sums of attenuations**
> `a_e = −ln(1 − 2p_e)`, a linear system that can be solved for individual `p_e`. The Spitz
> formula above is the two-detector special case.

## Boundary (single-detector) events

An event that flips only **one** detector `k` (e.g. an error at a spatial/time boundary, possibly
with `ℓ_e = 1` if it touches the logical) cannot be read from a pair correlation. It is found by
**hierarchical subtraction**: once all bulk edges at `k` are known, solve

```
1 − 2 p_k = ⟨z_k⟩ / Π_{edges (k,j)} (1 − 2 p_{kj})   with   ⟨z_k⟩ = 1 − 2⟨x_k⟩
```

Example: `⟨x_k⟩ = 0.050` so `⟨z_k⟩ = 0.900`; if two bulk edges at `k` have `p = 0.0034, 0.0030`,
then `Π(1−2p) = 0.98722`, giving `1 − 2p_k = 0.900 / 0.98722 = 0.91165`, so `p_k ≈ 0.0442`.
When marginals are small or correlators go negative (e.g. strong coherent ZZ crosstalk on
`ibm_miami`), this subtraction can return an **unphysical negative** `p_k` — Brown et al. clamp
those to zero.

## Side-by-side for the example edge `i–j`

| | Conventional (Path A) | Syndrome (Path B) |
|---|---|---|
| `h_e` | `{i, j}` from Stim propagation | `{i, j}` inherited from reference DEM |
| `ℓ_e` | `0` from Stim propagation | `0` inherited |
| Inputs to `p_e` | fault rates `r_f` (benchmarked) | `⟨x_i⟩, ⟨x_j⟩, ⟨x_i x_j⟩` (from shots) |
| Formula | `p_e = [1 − Π(1−2r_f)]/2` | Spitz: `p_ij = ½ − ½√(1 − 4·Cov/denom)` |
| Value | `0.0030` | `0.0034` |
| Captures device drift / non-Pauli? | No (only what the model encodes) | Yes (whatever the data shows) |
| Cost | Separate calibration circuits + model | None beyond the QEC run itself |

The two probabilities are close but not identical: the conventional value reflects the *assumed*
noise model, while the syndrome value reflects *what actually happened* on the device — including
drift, context dependence, and non-Pauli effects the static model omits. That gap is exactly the
~5–10% logical-error improvement Brown et al. report when swapping `p_e` for syndrome-estimated
values while keeping `(h_e, ℓ_e)` fixed.

*(Numbers above are illustrative, chosen to show the arithmetic; they are not taken from any
specific subsystem in the papers.)*

---

# Appendix A — Why the parity rule has those factors of 2

The combination rule

```
p_e = [ 1 − Π_f (1 − 2 r_f) ] / 2
```

is **not** simple addition `r₁ + r₂ + …` because a detector flips only when an **odd number** of
the contributing faults occur (the flip is an XOR / parity), and parities combine
*multiplicatively*. Both factors of 2 come from converting a probability into a ±1 "bias" and
back.

**1. Probability → bias (the `2` in front of `r_f`).** For one fault with probability `r`, let
`b = 1` if it fired. Its expected sign is

```
E[(−1)^b] = (1 − r)(+1) + r(−1) = 1 − 2r
```

So `1 − 2r` is just the "expected sign" of that fault (`r = 0` → bias `+1`; `r = ½` → bias `0`,
i.e. fully random). The factor of 2 is the map *probability → sign*.

**2. Parities multiply.** The detector flip is `b = ⊕_f b_f`, so its sign is the product of the
individual signs:

```
(−1)^b = Π_f (−1)^{b_f}   ⇒   E[(−1)^b] = Π_f (1 − 2 r_f)
```

**3. Bias → probability (the `÷ 2`).** The combined event has flip probability `p_e`, and by the
same rule its bias is `1 − 2 p_e`. Equate the two:

```
1 − 2 p_e = Π_f (1 − 2 r_f)   ⇒   p_e = [ 1 − Π_f (1 − 2 r_f) ] / 2
```

The `÷ 2` is the inverse map *bias → probability*, `p = (1 − bias)/2`. Same transform applied
forward to each `r_f` and inverted once for `p_e`.

**Sanity check (two faults).** The detector flips iff exactly one occurs:

```
p_e = r₁(1 − r₂) + r₂(1 − r₁) = r₁ + r₂ − 2 r₁ r₂
```

and the formula gives `[1 − (1 − 2r₁)(1 − 2r₂)]/2 = r₁ + r₂ − 2 r₁ r₂`. ✓
It is *not* `r₁ + r₂`: the `− 2 r₁ r₂` correction is because if **both** faults fire they cancel
and the detector does not flip.

This is why the natural variable is the **attenuation** `a = −ln(1 − 2p)`: the product becomes a
plain sum, `a_e = Σ_f a_f`, and the factors of 2 disappear (Young & Blume-Kohout's framing).

---

# Appendix B — What "`h_e`, `ℓ_e` inherited from a reference DEM" means

A DEM event's triple `(p_e, h_e, ℓ_e)` is built from **two independent inputs**, which it is
important not to conflate:

| Input | What it provides | Depends on error rates? |
|---|---|---|
| Noiseless **Clifford circuit analysis** (Stim error propagation) | the **support** — which events exist, and their signatures `(h_e, ℓ_e)` | **No** — pure geometry/combinatorics |
| **Device noise model** | physical fault rates `r_f` → `p_e` via the parity rule | **Yes** |

A "reference DEM" (SI1000, IBM-like, or RL-optimized) bundles both: Stim propagation builds the
support, and the noise model (or RL) fills in the probabilities.

**The syndrome-measurement path reuses only the first input.** Concretely (Brown et al.):

1. Take a reference DEM and **keep its support** `{(h_e, ℓ_e)}` and hyperedge decomposition.
   This is the Clifford-derived structure, which any decoder needs regardless.
2. **Discard** the reference `p_e` values (the noise-model part).
3. For each event, use its detector-flip pattern `h_e` to pick the matching empirical statistic
   and **re-estimate `p_e` from the syndrome data** — a bulk edge `h_e = {i,j}` from the `(i,j)`
   correlator, a boundary event `h_e = {i}` from the marginal via subtraction.

So the common phrasing "`h_e, ℓ_e` inherited" means: **the *circuit-structure* part is reused; the
device noise model's *probabilities* are thrown away and replaced by data-estimated `p_e`.** The
device model does **not** need to be accurate — only the *list of events and what they flip*
(which comes from the noiseless circuit, not from benchmarking) is reused.

**Two clarifications:**

- This is a deliberate *choice* in Brown et al., made to cleanly separate "rate estimation" from
  "choice of support." It is **not** fundamental. Remm et al. and Young & Blume-Kohout show the
  support itself can be *discovered from data* (which detectors are correlated ⇒ infer the
  edges), so a fully data-driven support is possible too.
- `ℓ_e` (does the event flip the logical?) is always a property of the **code/circuit geometry**
  — the known support of the logical operator — never something obtained by benchmarking. Even a
  fully data-driven pipeline gets `ℓ_e` from circuit structure, not from noise calibration.
