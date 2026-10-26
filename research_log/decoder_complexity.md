# Decoder complexity: why matching needs graphlike edges, and MWPM runtime (2026-10-24 → 2026-10-26)

Notation follows `mwpm_gap_derivation.md`.

## 1. Why MWPM is restricted to graphlike edges

**What is not the reason.** H′ is sparse, and BP on hyperedges is cheap. Measured on the d=3 DEM (column weight = number of detectors a mechanism flips):

| r | detectors | columns | column weights (weight: count) | nnz | max row weight |
|---|---|---|---|---|---|
| 3 | 24 | 219 | 1: 24, 2: 82, 3: 76, 4: 37 | 564 | 48 |
| 25 | 200 | 3056 | 1: 296, 2: 1091, 3: 1004, 4: 665 | 8150 | 64 |

At r=25 that is about 1.3% density. One BP sweep costs O(nnz(H′)).

**The reason: the complexity of each decoding problem.**

| problem | graphlike H (column weight ≤ 2) | general H (column weight ≥ 3) |
|---|---|---|
| min-weight error: min w·e subject to H′e = s (MWPM, Tesseract) | **P**. Solutions are T-joins with T = defects. A min T-join reduces to min-weight perfect matching on shortest-path distances, which Edmonds' blossom solves in O(\|T\|³) (sparse blossom is near-linear in practice). | **NP-complete** (Berlekamp, McEliece, van Tilborg 1978), by reduction from 3-dimensional matching, i.e. perfect matching on a 3-uniform hypergraph. |
| max-likelihood class: Σ over C_L(s) (2^{dim ker H′ − 1} = 2^261 terms at d=3, r=3) | hard in general (3D space-time graph) | **#P-hard** for stabilizer codes (Iyer & Poulin 2015) |
| BP marginals | cheap, inexact | cheap, inexact |

- **The precise cut:** perfect matching is in P on a graph and NP-complete on a hypergraph with weight-3 edges.
- **The hardness results are worst case.** They don't prove that the surface-code DEM in particular is hard, but no polynomial exact algorithm for it is known. They also don't depend on p. Tesseract's A* is practical at p ≤ 1e-3 because low-weight explanations dominate.
- The two citations come from Claude's background knowledge. They are not in `research_papers/` yet.

**Why BP is inexact.**
- BP gives exact marginals only on a tree. H′ has short cycles: e.g. the hyperedge D3 D4 D7 D8 shares two detectors with the edge D3–D4.
- **Degeneracy is the failure mode.** For a low-weight g ∈ ker H′, the errors e and e ⊕ g have the same syndrome and similar probability. The marginals on supp(g) then come out near ½, and the hard decision often fails H′ê = s.
- So BP alone has no threshold on the surface code (`project_docs_context/MWPM.md:122`). Belief-matching uses BP's posteriors only as matching weights.

## 2. From p to weights in general (MWPM, reweighting, ML decoders)

Let E(c) = −ln Q(c) be the energy of a graph-space correction c. Any positive distribution on {0,1}^M has a unique expansion:

  E(c) = K + Σ_i θ_i c_i + Σ_{i<j} J_ij c_i c_j + (higher order)

The optimal decoder computes:

  Λ*(s) = softmin_{C_1(s)} E − softmin_{C_0(s)} E,  softmin(E) = −ln Σ exp(−E)

**MWPM makes two static cuts:**
- **Model cut:** it drops J and every higher-order term, and sets θ_i = w_i = ln((1−p′_i)/p′_i), where p′_i is the marginal of Q. The product of marginals is the M-projection, argmin KL(Q‖·) over independent models. It depends only on p and never on s. → this is the correlation term.
- **Rule cut:** softmin → min. → this is the degeneracy term.

**Uniform circuit p does not give uniform edge weights.** Edges merge different numbers of faults, (1−2p′) = Π(1−2p_j). Example: the D7→boundary edge has p′ = 2.26e-3 (§4 of `mwpm_gap_derivation.md`). Only code capacity gives equal weights, which makes MWPM the same as minimum Hamming weight.

**Dynamic weights.** The exact weight of edge a, given all the other edges, is its local log-odds:

  w_a^eff = θ_a + Σ_b J_ab c_b + …

Decoders that reweight estimate the c_b from s:

  w_a(s) = θ_a + Σ_b J_ab ĉ_b(s)

**One hyperedge h with pieces a and b** (r_a, r_b = probability from each piece's other sources):
- θ_a ≈ ln(1/r_a), and J_ab ≈ −ln(p_h/(r_a r_b)) ≈ −ln(1/p).
- If b fired: w_a ≈ ln(r_b/p_h), close to 0.
- If b didn't fire: w_a ≈ ln(1/r_a).
- MWPM uses ln(1/(r_a+p_h)) in both cases.

**The ladder** (`project_docs_context/MWPM.md` §6):

| decoder | estimate of ĉ_b(s) | rule |
|---|---|---|
| MWPM | none (J = 0) | min |
| correlated matching | hard 0/1 from a first matching pass | min |
| belief-matching | BP posterior marginals on the hypergraph | min |
| ideal | exact posterior | softmin |

**AlphaQubit-style ML decoders don't output weights.**
- They output q_θ(L|s), trained on cross-entropy. The minimizer of that loss over the whole data distribution is the true posterior, so the network's log-odds converge to Λ*.
- J, the higher-order terms and the softmin are all implicit, learned only as far as they affect Λ.
- The network never sees p. Its effective energy comes from (s, L) data, so on hardware it can capture noise that no DEM contains.
- A hybrid that predicts w_θ(s) and feeds MWPM is a different design. Its loss has to pass through the non-differentiable min.

## 3. MWPM runtime: when it is fast and when it is slow

Reproduce with `code/claude/mwpm_runtime.py`.

**How sparse blossom works.** Each defect grows a region at the same rate.
- **Region hits the boundary or another free region:** the defect matches. O(1) local work.
- **Region hits an already-matched defect:** an alternating tree forms. The + nodes grow and the − nodes shrink.
- **Two + regions meet:** the augmenting path between them flips every match along it, at a cost proportional to its length.
- **Odd cycle:** a blossom forms, is contracted, and is later expanded.
- Cost scales with how far the effects of one defect's decision spread, not with the size of the graph.

**d=3, Z sector, one round** (chain Z0–Z2–Z1–Z3 joined by d3, d4, d5; every check has a boundary edge):
- {Z0}: one event, reaching the boundary through d6.
- {Z0, Z2}: one event, meeting on d3.
- {Z0, Z2, Z1}: Z0 and Z2 match first. Z1 then hits Z2, so a tree forms: Z1(+), Z2(−), Z0(+). Reaching the boundary rewires an existing match.
- Every check is one edge from the boundary, so cascades are at most length ~2. d=3 is always fast.

**Random circuit noise** (rotated memory-Z, r = d, average per shot):

| | p=1e-3 | 3e-3 | 6e-3 | 1e-2 |
|---|---|---|---|---|
| d=3: average defects, time | 0.3, 0.04 µs | 0.9, 0.08 µs | 1.7, 0.15 µs | 2.6, 0.26 µs |
| d=11: average defects, time | 22, 2.6 µs | 65, 10 µs | 124, 27 µs | 193, 63 µs |
| d=11: µs per defect | 0.10–0.18 | 0.14–0.17 | 0.22–0.27 | 0.35 |

- **At fixed p, time is roughly linear in the number of defects k.** The top 1% of shots by k take about 2× the median.
- **The cost per defect rises with p.** Denser defects mean bigger clusters, so bigger trees and longer augmenting paths. Near and above threshold the clusters percolate.

**Adversarial syndrome** (line graph, 20,001 nodes, unit weights, a contiguous block of k defects):

| k | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|
| even block | 8 ms | 36 ms | 148 ms | 594 ms (~k²) |
| odd block | 49 ms | 368 ms | 5.1 s | 62.6 s (~k³) |

- **Even block:** ties with both neighbours produce alternating trees along the whole block.
- **Odd block:** one defect must go to a boundary 10⁴ edges away. The whole block's tree is dragged along, and the final augmenting path rewires every pair. This is the Edmonds O(k³) behaviour.
- **For the FPGA tier:** a real-time decoder must meet its deadline on every shot, so the tail matters, not the mean (`project_docs_context/MWPM.md:97`). Union-find and Micro Blossom's parallel dual (`code/claude/micro_blossom_viz.rs`) exist to bound that tail.

## 4. The exact decoder in `mwpm.ipynb` is exponential in detectors, not errors

- **It doesn't enumerate errors** (2^M = 2^219 at r=3). It is dynamic programming over the joint table of (s, L) outcomes, at O(M · 2^(n+1)) cost: 219 × 2^25 ≈ 7×10⁹ operations, about 10 s.
- **The max (MLE) and sum (ML) folds cost the same.** Each produces a lookup table over all 2^24 syndromes at once.
- **It is exponential in the number of detectors n.** At r=25 there are 200 detectors, i.e. 2^201 entries.
- **At scale, use a per-syndrome search** (Tesseract) as the near-optimal proxy.
- graph-max computed this way only checks PyMatching, which solves that same problem in polynomial time.

## 5. Boundary vs bulk edges (2026-10-20)

- **MWPM has no preference for boundary edges.** It minimizes total weight, and a weight depends only on that edge's p′.
- **{Z0} → X_d6 is not a bias.** The alternative X_d0 X_d3 has weight 2: p vs 2p². Every decoder fails on it, because t = 1 at d=3.
- **Real effect 1: parallel-edge merging.** Z2's boundary edges through d0 and d1 merge to p′ ≈ 2p, about ln 2 cheaper. The same holds for Z1 through d7 and d8.
- **Real effect 2: path multiplicity.** This is degeneracy, and §7 of `mwpm_gap_derivation.md` puts it at ≤8% of the gap at d=3.
