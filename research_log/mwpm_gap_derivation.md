# Why MWPM picks the wrong fault class (2026-10-08 → 2026-10-13)

## 1. How PyMatching builds its graph

- **Stim computes the signatures.** It propagates every fault to the end of the circuit and records which detectors and observables flip. Each becomes a DEM line, e.g. `error(p) D3 D7 L0`. Faults with identical signatures are merged.
- **Each line becomes an edge.**
  - A line with 2 detectors becomes an edge between those two detectors.
  - A line with 1 detector becomes an edge to a single virtual boundary node.
  - The edge weight is w = ln((1−p)/p).
  - The L0 bit is stored on the edge as `fault_ids`. The prediction is the XOR of `fault_ids` along the matched edges.
- **The boundary node can be matched any number of times.** Physically, an error on a data qubit at the patch edge touches only one stabilizer of a given type. L0 is a line of data qubits from one logical boundary to the other, so the boundary edges on that side carry L0.
- **Lines with more than 2 detectors (hyperedges):**
  - Without decomposition, PyMatching **drops them silently**. This was checked: `error(0.01) D3 D4 D7 D8 L0` simply disappears.
  - With `decompose_errors=True`, Stim splits each one into pieces that already exist as edges on their own, e.g. `D3 D4 ^ D7 D8 L0`. A 3-detector error usually splits into a bulk edge plus a boundary edge. If no split is possible, it's a decomposition failure; with `ignore_decomposition_failures=True` that line is dropped.
- **Each piece gets the full p_h.** The probability isn't split between pieces. Each piece is merged into the existing edge for its detector pair:
  - (1 − 2p′_k) = Π_{sources j of k} (1 − 2p_j)
  - Checked: 0.01 combined with 0.01 gives 0.0198, so w goes from 4.60 to 3.90.

**How much of the DEM is hyperedges** (d=3, r=3, p=1e-3, share of total error probability):

| type | share |
|---|---|
| graph edge, within 1 round | 56% |
| graph edge, across 2 rounds | 32% |
| hyperedge, within 1 round | 4% |
| hyperedge, across 2 rounds | 8% |

Edges that cross rounds (timelike edges from measurement errors, and diagonal edges from errors partway through the CNOT schedule) are already in MWPM's 3D graph. Learning correlations over time is not an advantage over MWPM in itself. What matters is the hyperedges.

## 2. Spaces and terms

```
error space e ∈ F₂^M  ──H′──▶  detector space s = H′e  ──decoder──▶  L̂
(hidden)                       (= syndrome = detection events)
```

- **Detector:** D_{t,k} = m_{t,k} ⊕ m_{t−1,k}. The first and last rounds compare against the known initial state and against the final data readout.
  - This time-XOR is part of how a detector is defined. It is not H′.
  - Stim applies it when sampling, so `data_pipe`'s `events` channel is already s.
  - Its `cumxor` channel recovers the raw m.
- **H′ is not applied to data.** H′ is the fixed linear structure. The probabilities p_j are the statistics. The generating model is (H′, p):

  P(s, L) = Σ_{e : H′e = s, ℓ′e = L} Π_j p_j^{e_j} (1−p_j)^{1−e_j}

- **L0** is the parity of the final Z readout along one line of data qubits; it's the logical value. It isn't visible in s, so it serves as the label `y`. Degeneracy means the same s can occur with either value of L.
- **Detectors in a DEM don't depend on earlier errors.** Errors combine by XOR. A lingering X error makes a detector fire once, when it appears. It doesn't change how later noise acts. Errors that do depend on the qubit's state, such as leakage, exist only on real hardware.

## 3. The algebra (circuit-level DEM)

**Setup.**
- True mechanisms are the decomposed DEM lines j = 1..M.
- Graph edges are k = 1..M_G.
- The decomposition map is D ∈ F₂^{M_G×M}, with D_kj = 1 if piece k is in line j.

**Factorization** (verified numerically by `code/claude/kernel_dims.py`):

  H′ = H_G D,  ℓ′ = ℓ_G D,  D surjective  ⇒  dim ker H′ = dim ker D + dim ker H_G

| d=3 | columns of H′ | dim ker H′ | columns of H_G | dim ker H_G | dim ker D |
|---|---|---|---|---|---|
| r=1 | 29 | 21 | 23 | 15 | 6 |
| r=3 | 286 | 262 | 78 | 54 | 208 |

(Stim's undecomposed DEM has 219 columns at r=3, because it merges lines that decompose differently, e.g. `D3 D7` with `D3 L0 ^ D7 L0`.)

**Lemma 1: going to graph space loses no class information.**
- D preserves s and L, and it's surjective, so D(C_L(s)) = C^G_L(s).
- With the pushforward Q = D_*P:

  P(C_L(s)) = Σ_{g ∈ C^G_L(s)} Q(g)

- Comparing kernel dimensions therefore can't prove anything about the gap. At r=1, dim ker H = dim ker H_G = 15 for the undecomposed DEM, yet MWPM is still suboptimal there.

**Lemma 2: MWPM's model is the product of Q's marginals.**
- P_G(g) = Π_k p′_k^{g_k} (1−p′_k)^{1−g_k}, with p′_k = Q(g_k = 1).
- P_G = Q iff there are no hyperedges.

**Decision rule and splitting the error.**
- Optimal: Λ* = ln P(C_0) − ln P(C_1).
- MWPM: Λ̂ = max_{C^G_0} ln P_G − max_{C^G_1} ln P_G.
- MWPM is wrong exactly when the two have opposite signs.
- Let Λ^Q_max = max_{C^G_0} ln Q − max_{C^G_1} ln Q. Then:

  Λ* − Λ̂ = [Λ* − Λ^Q_max] + [Λ^Q_max − Λ̂]

  The first bracket is the **degeneracy** term (sum vs max). The second is the **correlation** term (Q vs ΠQ_k).

**Size of the correlation term for one hyperedge.** Take hyperedge h with pieces a and b, where r_a is the probability from a's other sources:
- **a without b:** ln Q − ln P_G ≈ ln[r_a / (r_a + p_h)] < 0. MWPM overrates it.
- **a and b together:** ln Q − ln P_G ≈ ln[p_h / ((r_a + p_h)(r_b + p_h))] ≈ +ln(1/p). MWPM underrates it.
- MWPM gets each edge's own probability right. It gets the joint wrong, and the joint is what it optimizes over.

## 4. Worked example: the syndrome where MWPM loses most (d=3, r=3, p=1e-3)

**Where the gap sits.**
- The total gap is 1.598e-4 (= 7.699e-4 − 6.101e-4), spread over about 600k syndromes.
- The single largest syndrome is s = {D2, D7}. All three detectors involved are Z-type:
  - D2 = (4,4), round 0
  - D3 = (6,2), round 0
  - D7 = (6,2), round 1

**Exact values.**
- P(s,0) = 6.09e-6 and P(s,1) = 2.36e-6.
- So Λ* = +0.95: L=0 is correct, with P(L=1 | s) = 0.28.

**MWPM's two candidate paths:**

| path | edges | L | MWPM cost Σw′ | true cost −ln Π o_j (leading term of −ln Q) |
|---|---|---|---|---|
| A | D2–D3 + D3–D7 (measurement error) | 0 | 11.99 | 12.27 |
| C | D2→boundary + D7→boundary | 1 | 11.30 | 12.82 |

- MWPM picks C (Λ̂ = −0.69). It's wrong 72% of the times this syndrome appears.

**Why C looks cheap.**
- The D7→boundary edge has p′ = 2.26e-3, but only 6.67e-4 of that comes from a genuine "D7 alone" error.
- The rest comes from 11 hyperedge pieces, such as `D6 ^ D7 L0` and `D14 ^ D7 L0`. All of their partner detectors (D3, D6, D13, D14, D15) are absent from s.
- Correction: ln(0.667/2.26) = −1.22. The D2 boundary edge adds ln(4.05/5.44) = −0.30.

**Splitting the error:**
- Λ^Q_max = 12.82 − 12.27 = +0.55
- correlation term = 0.55 − (−0.69) = 1.24, which **flips the sign by itself**
- degeneracy term = 0.95 − 0.55 = 0.40

Reproduce with `code/claude/worst_syndrome.py`. The true costs use the decomposed DEM lines as the mechanisms, so that each one's image in the graph is known.

## 5. Clean proof (code-capacity model: 9 data qubits, perfect measurement)

**Matrices.**
- H_G = [H_x 0; 0 H_z] (8 × 18)
- H_true = [H_x 0 H_x; 0 H_z H_z] (8 × 27). The third block is the Y errors.
- D = [I 0 I; 0 I I]
- Both have rank 8, and dim ker = 10 and 19.
- ker D is spanned by X_i + Z_i + Y_i, so 19 = 9 + 10.

**Noise.** Under depolarizing noise, X, Y and Z on a qubit are mutually exclusive.
- An equivalent independent model gives each of them q, with (1−2q)² = 1 − 4p/3. This is exact and is what Stim uses.
- The cleaner route is to change variables to the net Pauli, x = e_X + e_Y and z = e_Z + e_Y. The per-qubit joint is then:

| | z_i=0 | z_i=1 |
|---|---|---|
| **x_i=0** | 1−p | p/3 |
| **x_i=1** | p/3 | p/3 (Y) |

  P(s, L) = Σ_{x : H_x x = s_x, ℓx = L} Σ_{z : H_z z = s_z} Π_i q(x_i, z_i)

**Factoring out the correlation.**
- Write q(x_i, z_i) = q_x(x_i) q_z(z_i) c(x_i, z_i), with marginals q(1) = 2p/3. Those marginals are MWPM's edge probabilities.
- The correlation factor is c(1,1) = 3/(4p), c(1,0) = c(0,1) ≈ 1/2, c(0,0) ≈ 1 + p/3.
- Then:

  P(s, L) = Σ_{x ∈ C_L(s_x)} P_x(x) · R(x; s_z),  R(x; s_z) = Σ_{z : H_z z = s_z} P_z(z) Π_i c(x_i, z_i)

**Comparison.**
- **Optimal:** argmax_L Σ_{x ∈ C_L(s_x)} P_x(x) R(x; s_z)
- **MWPM:** the L of argmax_{x : H_x x = s_x} P_x(x). H_G is block-diagonal and the weights add, so the problem splits into two separate ones and **L̂ depends on s_x only**.
- MWPM sets R ≡ 1, which drops s_z (the correlation term), and takes the max instead of the sum (the degeneracy term).
- R ≡ 1 iff q(1,1) = q_x(1) q_z(1); with depolarizing noise, that means iff p_Y = 0. So MWPM ≠ ML whenever p_Y > 0.
- **Reading R:** a candidate x with an error on qubit i gains a factor 3/(4p) (≈ 250× at p=1e-3) if the Z sector also shows a flip there. It loses a factor 2 if the Z sector is quiet.
- **Correlated matching** approximates R: decode Z, reweight the X edges with P(X | Z), decode X again.

## 6. What this means for the ML decoder

- **Inside a DEM:** the only thing the model can learn that MWPM can't is the conditional structure R. That means co-located X and Z detection events, and hyperedge patterns spanning two rounds. At d=3 this is worth 16–21%.
- **On real data:** effects like leakage cause bursts of detection events over many rounds, until the qubit is reset. This needs memory across rounds, through recurrence or temporal graph edges. AlphaQubit keeps its lead up to d=11 on Pauli+ noise (`decoder_ML.md:88`) and takes leakage flags as input (`decoder_ML.md:159`).

## 7. The split over all syndromes (2026-10-20)

Code: the last cell of `code/mwpm.ipynb`. Four exact decoders sit on a (model × rule) grid: graph or true DEM, max or sum. All are scored on the true P(s, L) at d=3, r=3.
- The max rule uses the same XOR fold with `max` in place of `+` (max-product), so it is exact.
- graph-max reproduces PyMatching on every syndrome (mismatch mass 0).
- The shares below are the Shapley average of the two orderings. The ordering moves them by at most ±2 points.

| p | MWPM ÷ opt | true-max ÷ opt | graph-sum ÷ opt | correlation | degeneracy |
|---|---|---|---|---|---|
| 1e-3 | 1.262 | 1.013 | 1.253 | 96% | 4% |
| 2e-3 | 1.241 | 1.014 | 1.231 | 95% | 5% |
| 3e-3 | 1.222 | 1.016 | 1.211 | 94% | 6% |
| 5e-3 | 1.189 | 1.019 | 1.178 | 92% | 8% |

- **Fixing correlations alone closes 90–95% of the gap.** A most-likely-error decoder on the hypergraph is within 1.3–1.9% of optimal.
- **Summing over the class on MWPM's own graph barely helps** (1.262 → 1.253 at p=1e-3).
- **Per syndrome** (weighted by |P(s,0) − P(s,1)|, where MWPM differs from optimal):
  - fixed by correlation alone: 84–89%;
  - fixed by either fix: 10–13%;
  - fixed by degeneracy alone: ≤1%;
  - fixed only by both together: 1–3%.
- **The degeneracy share grows with p** (more near-tied classes). It is still small at p=5e-3.
- **This differs from §4's worst syndrome** (1.24 vs 0.40 in log-likelihood units). §4 measures the log-odds shift on one syndrome; this measures LER summed over all of them.
- **For the ML decoder:** the target is the hyperedge/conditional structure (R in §5). Path multiplicity is not. Boundary-vs-bulk path counting falls under degeneracy, so it is a ≤8% effect at d=3.
