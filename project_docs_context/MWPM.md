# Minimum-weight perfect-matching (MWPM) decoding: concept, cost function, limits

Study note, 13 October 2026. Source for §1–§3: Y. Kim, *Matching decoder — how it works?* (OSU slides, 13 Sep 2026; [`./QEC/MWPM_slides.pdf`](./MWPM_slides.pdf)). §4–§6 place matching inside the detector-error-model (DEM) picture (see `DEM.md` and `DEM_prior.md` in this folder) and survey what matching cannot do and what comes next.

---

## 1. From syndromes to a graph

**Stabilizers and syndromes.** A stabilizer code measures a set of commuting parity checks $S_a$ without collapsing the logical state. For the 3-qubit repetition code the checks are $Z_1Z_2$ and $Z_2Z_3$; the code space $\{|000\rangle,|111\rangle\}$ gives outcome $(+,+)$, and a single bit flip moves the state into a detectable error space ($|010\rangle$: both checks fire; $|100\rangle$: only the first). The syndrome $s$ is the vector of check outcomes; **decoding** is the map $s\mapsto$ a correction.

**Detectors.** In a repeated-measurement experiment the stabilizers are measured every round. A measurement outcome is not by itself informative (it depends on the prepared state and on everything before); the informative quantity is the *change* between rounds,
$$D(n)=S(n-1)\oplus S(n),$$
which is deterministically $0$ in the absence of faults. Each $D(n)$ is one **detector**; the experiment produces one detector bit per check per round.

**Error mechanisms as edges.** Every physical fault flips some set of detectors. For the repetition code:

| fault | detectors flipped | edge type | mechanism bundle |
|---|---|---|---|
| $X$ on a data qubit between rounds | the two checks adjacent to that qubit, same round | *space-like* (horizontal in the slides) | 1Q data error, 2Q-gate error landing on data, $T_1$ decay, … $\;\to p_{e_1}$ |
| fault on an ancilla / measurement error | the same check in two consecutive rounds, $D(n)$ and $D(n+1)$ | *time-like* (vertical) | 1Q ancilla error, 2Q-gate error landing on the ancilla, readout error, … $\;\to p_{e_2}$ |
| data fault in the middle of the check circuit ("hook") | one check this round and the neighbouring check next round | diagonal | gate errors mid-cycle |
| fault on an edge data qubit | **one** detector only | *boundary edge* to a virtual vertex | same bundles as above |

The picture (detectors = vertices, mechanisms = edges, $p_e$ = total probability of the bundle) is the **detector graph**: the special case of a DEM (`DEM.md` §1) in which every mechanism $e$ flips a set $D(e)$ of at most two detectors, $|D(e)|\le2$. A DEM lists mechanisms $e$ with probability $p_e$, detector set $D(e)$ and logical action $\Lambda(e)$ (whether the fault flips the logical observable). The slides' key observation is that *every* error appears as a **pair** of flipped detectors — because a check measures a parity and a single fault changes parity at exactly two places — except at the lattice boundary, where one end of the pair is absorbed by a **virtual ancilla** ("add virtual ancilla → create pair"). After adding the virtual vertex, every fault is again an edge between two vertices.

**Surface code.** The same picture in 2D: data qubits on faces, ancillas on vertices measuring $X$- or $Z$-type plaquettes ("the syndrome measurement is designed to measure parity"). A single $Z$ error on a data qubit flips the two adjacent $X$-checks; a single $X$ error flips the two adjacent $Z$-checks; the two sectors are decoded as two independent matching problems. Multiple adjacent errors produce a longer chain whose only visible signature is its two endpoints (slide 21), and an error chain that ends on the boundary has one endpoint (slide 23–24).

## 2. The cost function

Let $M$ be a set of edges (a hypothesised error configuration). With independent edge faults, the probability of exactly that configuration is
$$P(M)=\prod_{e\in M}p_e\prod_{e\notin M}(1-p_e),$$
which *includes the probability that every other edge did not fire* (slide 12). Taking $-\log$ and splitting the second product,
$$-\log P(M)=-\sum_{e\in M}\log p_e-\Big[\sum_{\text{all }e}\log(1-p_e)-\sum_{e\in M}\log(1-p_e)\Big]
=\sum_{e\in M}\underbrace{\log\frac{1-p_e}{p_e}}_{w_e}\;-\;\underbrace{\sum_{\text{all }e}\log(1-p_e)}_{\text{same for every }M}.$$
Hence
$$\boxed{-\log P(M)=\sum_{e\in M}w_e+\text{const},\qquad w_e=\log\frac{1-p_e}{p_e}.}$$
A more probable configuration has a smaller total weight: **maximum probability $\Leftrightarrow$ minimum weight** (slide 14). For $p_e\ll1$, $w_e\approx\log(1/p_e)$, so an edge with probability $p$ costs $\approx\log(1/p)$ and a chain of $k$ edges costs $\approx k\log(1/p)$ — the weight counts faults, log-weighted by rarity.

**The decoding problem.** Given the observed set of fired detectors $\partial M$ (the "defects"), find
$$M^\ast=\arg\min_{M:\ \partial M=\text{observed defects}}\ \sum_{e\in M}w_e .$$
Slide 15 shows the canonical ambiguity: four fired detectors can be explained by two horizontal edges ($w=2\log\frac{1-p_{e_1}}{p_{e_1}}$) or two vertical edges ($w=2\log\frac{1-p_{e_2}}{p_{e_2}}$); the decoder picks the cheaper, i.e. the more probable mechanism (data error vs. measurement error).

**Reduction to perfect matching.** Because every edge has two ends, any valid $M$ pairs up the defects (each defect is the endpoint of an odd number of edges in $M$; interior vertices of a chain are touched twice). So $M$ decomposes into paths between pairs of defects, and the cheapest $M$ with given endpoints is a union of shortest paths. Build the complete graph on the defects (plus one virtual copy of each defect connected to the boundary), with
$$w(i,j)=\text{shortest-path weight from }i\text{ to }j\text{ in the detector graph,}$$
and solve
$$M^\ast=\arg\min_{M\ \text{a perfect matching}} C(M),\qquad C(M)=\sum_{(i,j)\in M}w(i,j),$$
with the **constraint that every defect is paired** ("if they do not match, that's not a valid configuration — except edge qubits", slide 25). This is Edmonds' blossom problem, polynomial in the number of defects. Sparse Blossom (PyMatching 2) solves it directly on the detector graph by growing alternating trees without building the all-pairs graph, at roughly a million errors per core-second — about $1\,\mu$s per round for a $d=17$ surface code under circuit noise ([Higgott & Gidney, Quantum 2025](https://quantum-journal.org/papers/q-2025-01-20-1600/)).

**What is actually output.** The correction is $\prod_{e\in M^\ast}P_e$, but only its *logical class* matters: two corrections differing by a stabilizer act identically. The decoder's answer is therefore the pair (frame flip or not); in DEM language it is $\Lambda(M^\ast)=\bigoplus_{e\in M^\ast}\Lambda(e)$.

## 3. What MWPM computes, and what it does not

MWPM returns the **most likely single error** (MLE, "most-likely-error") among error configurations compatible with $s$, under two assumptions:

1. every mechanism flips at most two detectors (**graphlike**, $|D(e)|\le2$);
2. mechanisms are independent, with probabilities $p_e$ that are **known** (they are the DEM rates; `DEM_prior.md` and [Blume-Kohout & Young 2025](https://arxiv.org/abs/2504.14643) cover how to estimate them from syndrome data).

The *optimal* decoder is different: it maximises the probability of the **logical class**, not of a single configuration,
$$L^\ast=\arg\max_L Z_L(s),\qquad Z_L(s)=\sum_{E:\,D(E)=s,\ \Lambda(E)=L}\prod_{e\in E}p_e\prod_{e\notin E}(1-p_e),$$
the sum over all fault sets $E$ in the *coset* of logical class $L$ (all $E$ reproducing $s$ with the same logical action). MWPM replaces each sum by its largest term. The two agree when one term dominates — low $p$, and no large *degeneracy* (many distinct configurations of the same minimal weight in one class). They disagree when the entropy of a class matters, which is why MWPM's threshold ($\approx10.3\%$ for code-capacity, $0.8\%$ for circuit-level depolarizing) is below the ML threshold ($\approx10.9\%$, $\approx1\%$) ([Bravyi, Suchara, Vargo 2014](https://arxiv.org/abs/1405.4883); [Tobias, Breuckmann, Placke 2026](https://arxiv.org/abs/2603.05428)).

## 4. Two structural limits of matching

### 4.1 $|D(e)|\le2$: the graph assumption

A matching graph can only hold mechanisms that flip at most two detectors. The circuit-level DEM of the surface code is a **hypergraph**:

- A **$Y$ error** on a data qubit flips two $X$-type and two $Z$-type detectors — four in total. Matching handles it by *decomposing* it into one $X$-sector edge and one $Z$-sector edge, and then decoding the two sectors independently. Two things are lost: the probability bookkeeping is wrong (the $Y$ rate is added to both edges as if they were independent), and the **correlation** is discarded — knowing that the $X$ sector matched through qubit $q$ should raise the prior that the $Z$ sector also runs through $q$, but the independent decoder never learns it. Under depolarizing noise a third of all faults are $Y$'s, so this is not a small effect: it costs roughly $15\%$ in threshold and a factor of a few in logical error rate at low $p$ ([Higgott et al. 2023](https://arxiv.org/abs/2203.04948)).
- **Hook and gate faults** in the syndrome-extraction circuit flip up to four detectors as well; some decompose into existing edges, some only approximately.
- **Correlated logical operations.** A transversal CNOT between two code blocks propagates faults across blocks, so one mechanism flips detectors in both blocks; decoding the blocks separately throws that away ([Cain et al. 2024](https://arxiv.org/abs/2403.03272)).
- **Non-matchable codes.** In the 2D colour code a single fault flips three detectors; in bivariate-bicycle / qLDPC codes checks have weight 6 and more and each fault touches several detectors. No decomposition into a graph exists; matching does not apply at all.

The DEM remains a valid object in all these cases — only the decoder's *view* of it is truncated to its graph part. The coset structure of the true hypergraph model differs from that of the decomposed graph model, so any failure probability inferred from the matching graph is biased.

### 4.2 No ties: matching is confidently wrong and carries no confidence

For a syndrome $s$ let $W_0(s)$ and $W_1(s)$ be the minimal weights (number of faults, for uniform $p_e$) of the cheapest identity-class and cheapest logical-class explanations of $s$. In a decoupled $X$-only or $Z$-only matching graph of odd distance $d$, any two explanations of the same syndrome from different classes differ by a logical operator, whose weight is $\equiv d\pmod2$; hence
$$W_0(s)+W_1(s)\equiv d\equiv1\pmod2\quad\Longrightarrow\quad |W_1(s)-W_0(s)|\ \text{is odd},\ \ge1,\ \text{for every }s .$$
Three decoder-level consequences follow.

- **The dominant failure is a weight-1 margin.** A fault configuration of weight $(d+1)/2$ is explained more cheaply by a weight-$(d-1)/2$ configuration of the other class; MWPM applies that correction and the product is a weight-$d$ logical operator. The two explanations differ by exactly one fault.
- **Matching never hesitates.** There is no syndrome on which the two classes tie, so MWPM never flags a shot as undecidable; it fails by being *confidently wrong*, with a per-syndrome failure probability of order $p^{|W_1-W_0|}\le p$. The failures are spread over the large set of syndromes with margin 1 rather than concentrated on a few obviously bad ones, which is why plain MWPM provides no useful confidence signal for post-selection or for hierarchical codes (§5, "no soft output").
- **The absence of ties is an artefact of the graph model.** With $Y$ correlations (a $Y$ fault counts once, not once per graph), hook hyperedges, or even $d$, the parity argument fails: $W_0(s)=W_1(s)$ ties exist and the true posterior has syndromes with $\Theta(1)$ failure probability. A matching decoder reports a margin-1 landscape for a problem whose real landscape has ties, so it underestimates its own failure probability exactly on the shots that matter most. The *complementary gap* (§5) is the cheap way to recover a confidence from matching; whether that confidence is calibrated depends on how much of 4.1 has been fixed.

Points 4.1 and 4.2 are the same defect seen from two sides: truncating the hypergraph to a graph both mis-weights the error classes and erases the ties and degeneracies of the true problem.

## 5. Other major problems with matching (survey)

| problem | what goes wrong | representative fixes |
|---|---|---|
| **Degeneracy / entropy** | MWPM compares single configurations, not class probabilities $Z_L(s)$; at moderate $p$ the number of near-minimal paths matters. | ML decoding by tensor networks / MPS ([BSV 2014](https://arxiv.org/abs/1405.4883)); multi-path summation ([Criger & Ashraf 2018](https://arxiv.org/abs/1709.02154)); MCMC "worm" sampling of cosets ([Tobias et al. 2026](https://arxiv.org/abs/2603.05428)). |
| **Non-matchable codes** | colour codes and BB/qLDPC codes have $\lvert D(e)\rvert\ge3$ for *single* faults, so there is no graph to match on at all (details in §5.1). | code-specific projections to matching (restriction / Möbius decoders for colour codes); BP+OSD and its faster variants for qLDPC; search-based MLE on the hypergraph DEM (Tesseract). |
| **No soft output** | MWPM returns a class but no confidence; needed for post-selection, for hierarchical/yoked codes, and for any scheme that weights shots by decoder confidence. | **complementary gap**: run matching twice, once forced into the other logical class; $\text{gap}=C(M^\ast_{\bar L})-C(M^\ast_L)$ and the per-shot failure probability is $p_L(s)\approx(1+e^{\rm gap})^{-1}$ ([Gidney et al. 2023](https://arxiv.org/abs/2312.04522), [Meister et al. 2024](https://arxiv.org/abs/2405.07433)). |
| **Wrong or drifting $p_e$** | weights assume a calibrated DEM; miscalibration is a bias. | DEM estimation from syndrome data ([Blume-Kohout & Young 2025](https://arxiv.org/abs/2504.14643)); online reweighting. |
| **Leakage, analog readout, non-Markovian noise** | not representable as independent binary edges. | soft-information edge weights; leakage-aware DEMs; neural decoders trained on data (AlphaQubit-type). |
| **Boundaries** | the virtual-vertex trick is exact for CSS surface codes but tailored codes (XY, XZZX) have "fragile boundaries" where $O(\sqrt n)$ low-rate errors plus one high-rate error cause failure ([Higgott et al. 2023](https://arxiv.org/abs/2203.04948)). | rectangular CSS codes; boundary-aware weights. |
| **Throughput / latency** | decoding must keep up with $\sim1\,\mu$s cycles; blossom is fast but worst-case super-linear; backlog grows exponentially if it falls behind. | union-find (almost-linear), decoder switching between fast and accurate decoders ([2510.25222](https://arxiv.org/abs/2510.25222)), parallel windowing, FPGA/ASIC. |
| **Correlated logical algorithms** | transversal gates and lattice surgery couple blocks; independent decoding loses the cross-block hyperedges. | correlated decoding across blocks ([Cain et al. 2024](https://arxiv.org/abs/2403.03272)). |

### 5.1 Non-matchable codes in more detail

Matching needs every mechanism to flip at most two detectors. That holds for the surface code because each data qubit sits on exactly two $X$-checks and two $Z$-checks, so a single $X$ or $Z$ fault produces a *pair* of defects. It fails as soon as a data qubit belongs to three or more checks of the same type:

- **2D colour code.** Every data qubit is shared by three faces of each colour class, so a single $Z$ fault flips **three** $X$-type detectors (one per face); $\lvert D(e)\rvert=3$ already at the single-fault level and the detector "graph" is a 3-uniform hypergraph. The standard fix is to *project* onto matchable sub-problems: the **restriction decoder** ([Kubica & Delfosse 2023](https://arxiv.org/abs/1905.07393)) restricts the syndrome to two colour classes at a time, where faults again look like pairs, matches there, and lifts the result back; the **Möbius / Chromobius** decoder ([Sahay & Brown 2022](https://arxiv.org/abs/2108.11395); [Gidney & Jones 2023](https://arxiv.org/abs/2312.08813)) glues the three restricted graphs into one matching problem on a Möbius-strip-like graph so a single blossom call suffices. Both are heuristics: they pay $\sim$10–30 % in threshold relative to MLE and are the reason colour-code thresholds under circuit noise sit below the surface code's despite the better encoding rate.
- **Bivariate-bicycle and other qLDPC codes.** Checks have weight 6 (gross code $[[144,12,12]]$) or more and each data qubit is in several checks of the same type, so single faults flip 3–6 detectors; there is no two-colour structure to restrict to either. The workhorse is **belief propagation + ordered-statistics decoding (BP+OSD)** ([Panteleev & Kalachev 2021](https://arxiv.org/abs/1904.02703); [Roffe et al. 2020](https://arxiv.org/abs/2005.07016)): BP on the Tanner graph gives soft marginals; when BP fails (it does, because of short cycles and degeneracy), OSD orders the mechanisms by posterior, Gaussian-eliminates the most reliable set and completes the solution — accurate but $O(n^3)$ and slow. Faster variants reduce or localise the post-processing: BP+LSD (localised statistics, parallel clusters, [Hillmann et al. 2024](https://arxiv.org/abs/2406.18655)), ambiguity clustering ([Wolanski & Barber 2024](https://arxiv.org/abs/2406.14527)), relay-BP ([Müller et al. 2025](https://arxiv.org/abs/2506.01779)), each trading a little accuracy for real-time feasibility.
- **Exact or near-exact on the hypergraph.** Integer programming finds the most-likely error for any DEM but is exponential in the worst case; **Tesseract** ([Beni, Higgott, Shutty 2025](https://arxiv.org/abs/2503.10988), [Grbic et al. 2026](https://arxiv.org/abs/2602.02985)) does A* search over partial error sets with an admissible heuristic, pruning and beam cut-offs, matching integer-programming accuracy at $p\le10^{-3}$ on surface, colour and BB codes while running $\sim5\times$ faster, and 1–2 orders of magnitude lower logical error than BP+OSD on the BB codes tested. It is still most-likely-*error* (single configuration), not maximum-likelihood over classes, so the degeneracy problem of §3 remains.

The common thread: when $\lvert D(e)\rvert\ge3$ one either (i) finds a sub-structure where faults are pairs again and matches there, (ii) replaces matching by an iterative marginal estimator (BP) plus a repair step, or (iii) searches the hypergraph directly. Option (i) is specific to colour codes; (ii) is the generic qLDPC answer; (iii) is the accuracy reference the others are measured against.

## 6. The next-simplest decoder, and which problem it fixes

The least invasive upgrade that attacks limit 4.1 while keeping the matching solver is the **two-stage reweighting** family:

**(a) Correlated matching** (Fowler 2013; Google's "correlated sparse blossom"). Decode the $X$ sector first; wherever the matched $X$ chain passes through a qubit, lower the weight of the corresponding $Z$-sector edge to reflect that a $Y$ there would explain both; then decode the $Z$ sector. Cheap, captures the largest correlation, order-of-magnitude in nothing but a steady few-× gain in logical error at low $p$.

**(b) Belief-matching** ([Higgott, Bohdanowicz, Kubica, Flammia, Campbell 2023](https://arxiv.org/abs/2203.04948)) — the recommended next step:

1. Run **belief propagation** on the full circuit-level *Tanner graph* of the DEM (variables = hyperedge mechanisms, checks = detectors), producing a posterior marginal $\hat p_e(s)$ for every mechanism given the observed syndrome.
2. Push each hyperedge's posterior onto the edges of its decomposition and set the matching weights to $w_e(s)=\log\frac{1-\hat p_e(s)}{\hat p_e(s)}$ — the same cost function as §2, but now **syndrome-dependent** and informed by $Y$ and hook correlations.
3. Run MWPM on the reweighted graph (or weighted union-find: "belief-find", almost-linear time, same accuracy).

BP alone has no threshold on the surface code (it fails on degenerate loops), which is why it is used only to *reweight*. Results under circuit-level depolarizing noise: threshold $0.817\%\to0.940\%$ ($1.15\times$); under biased noise on the XY code $0.498\%\to0.841\%$, and at $p=0.27\%$ the logical error rate is $7.8\times$ lower than MWPM while still $2.8\times$ above the ML decoder.

Belief-matching fixes the **weight assignment** part of 4.1 but still outputs a single matched class, so it does not by itself fix 4.2; the hypergraph posterior it computes, however, is exactly the soft information that the complementary-gap trick (§5) needs, and combining belief-matching with a forced-class second pass gives a cheap, correlation-aware per-shot failure probability. For any use that weights or discards shots by decoder confidence, this combination (hypergraph-aware reweighting + complementary gap) is the smallest step that supplies both a better correction and a calibrated confidence.

Beyond this, the ladder is: Tesseract / integer-programming MLE on the hypergraph (fixes 4.1 completely, still MLE not ML); tensor-network or MCMC ML decoding (fixes degeneracy, expensive); learned decoders (fix model mismatch, need data).
## 7. Learned decoders: what they fix, and how to start

*Companion note: `decoder_ML.md` in this folder has the neural-decoder literature (AlphaQubit, AlphaQubit 2, RL control), the public datasets, and a build ladder. This section is the bridge — it says which of §4–§5's defects a learned decoder actually repairs, what the honest baseline is, and which implementation to start from.*

### 7.1 A naming trap, first

In this note and in the QEC literature generally, **"ML decoder" means *maximum-likelihood* decoder** — §3's $L^\ast=\arg\max_L Z_L(s)$, realised by tensor networks or MCMC (§5, degeneracy row). Searching for "ML decoder" returns Bravyi–Suchara–Vargo and the worm algorithm, not machine learning.

The machine-learning family is called a **neural decoder** (also *neural-network decoder*, *learned decoder*). Use that term; the collision is otherwise constant, and awkward precisely because MLD is the accuracy ceiling a neural decoder is measured against.

### 7.2 What a neural decoder fixes

A neural decoder takes the detector tensor directly and is trained on the logical label. Both of §4's structural limits dissolve as a consequence of that choice — they are not fixed by added machinery, they simply never arise.

| Limit in this note | Learned decoder | Why |
|---|---|---|
| **§4.1** hypergraph truncation, $Y$ correlations, hooks | **fixed by construction** | No graph is built, so there is no decomposition step to be lossy. Correlations are whatever the network learns from data. |
| **§4.2** no ties, no confidence | **fixed by construction** | Output is $P(\text{flip})\in[0,1]$. AlphaQubit's calibration result — 0.2 % rejection → ~20× error reduction — is §5's "no soft output" row answered without the forced-class second pass. |
| **§5** leakage, analog readout, non-Markovian | **strongest case** | Soft I/Q values and leakage flags are extra input channels. Matching has no natural slot for either; the best it manages is a per-measurement likelihood ratio folded into $w_e$. |
| **§5** wrong or drifting $p_e$ | partially | Replaces a hand-fitted DEM with learned structure, but substitutes a train/test mismatch for a calibration error. Does not remove the problem, relocates it. |
| **§5** degeneracy, class vs configuration | **fixed in principle** | Trained against the logical label, so the objective *is* $\arg\max_L Z_L(s)$ — ML, not MLE. Whether it reaches that optimum is an empirical question. |
| **§5** throughput / latency | **made worse** | AlphaQubit was far from 1 µs/round; AQ2-RT reaches it only on Trillium TPUs. This is the one axis where matching is ahead, and the one that matters for a real-time (FPGA) tier. |
| **§5.1** non-matchable codes | fixed | Nothing in the architecture cares whether $\lvert D(e)\rvert\ge3$. Colour and BB codes are the same problem to a network. |

The pattern is worth stating plainly: a learned decoder repairs both structural limits of §4 and most of the §5 survey, and degrades exactly one row — throughput.

### 7.3 The baseline problem

**Beating plain PyMatching establishes nothing.** §6 records that belief-matching alone moves the circuit-level threshold $0.817\%\to0.940\%$ and lowers LER by $7.8\times$ at $p=0.27\%$. Any learned decoder whose margin over plain MWPM is smaller than that has been beaten by one BP reweighting pass. A large share of the published neural-decoder literature compares against bare PyMatching and is, for this reason, uninformative.

The honest baseline set is therefore **correlated matching** and **belief-matching** (§6a, §6b), plus **gap-MWPM** (§5, soft-output row) whenever a calibration or post-selection claim is being made — matching does carry a confidence signal, cheaply, and it must be beaten rather than ignored.

A second caution, from a $d=3$ hardware experiment (IBM heavy-hexagon subsystem code) with every decoder run on identical shots ([Sundaresan, Yoder, Kim et al. 2023](https://doi.org/10.1038/s41467-023-38247-5)). Compared like for like — leakage-post-selected $\lvert+\rangle_L$, $r=2$ rounds, total logical error (Methods table "Logical error at $r=2$ rounds"):

| change | logical error |
|---|---|
| analytic matching → full MLD | $20.91\%\to20.83\%$ ($-0.08$ pp) |
| MLD, deflag every round → deflag first round only | $20.83\%\to18.02\%$ ($-2.81$ pp, ≈13 % relative) |

The preprocessing change is about $35\times$ the decoder upgrade. *Deflagging* turns flag-qubit outcomes into virtual Pauli-$Z$ corrections on data qubits before decoding, and drops the flags from the hypergraph. It is a hard decision, so a flag readout or reset error becomes a $Z$ error on a clean data qubit. MLD is exact over the hypergraph it is handed, so it cannot recover what was removed before that hypergraph was built. The deflagging result is in the published version only, not the arXiv preprint. Do not set it against the abstract's fitted per-round figures: matching vs MLD is $\approx0.040$ vs $0.037$ in $Z$ and $0.088$ vs $0.087$ in $X$, i.e. ~7 % and ~1 % *relative per round*, a different unit. All of it ran on leakage-post-selected data at 4.37–4.91 % rejection per round.

At small distance with real device noise, the decoder is not the bottleneck; the noise model, the preprocessing and the leakage handling are. Two consequences: do not make $d=3$ accuracy the headline claim, and measure the headroom (§7.6) before committing to the accuracy axis at all.

### 7.4 Where to start: an open implementation

The reference to build on, chosen because the architecture is fully specified *and* the code is released:

> **C. Chamberland, J. Olle, M. Li, S. Thornton, I. Baratta, *Fast and accurate AI-based pre-decoders for surface codes*, [arXiv:2604.12841](https://arxiv.org/abs/2604.12841) (2026).**
> Code: **<https://github.com/NVIDIA/Ising-Decoding>** (Apache-2.0).

**Why this one rather than AlphaQubit.** AQ1/AQ2 have no code or weights release (see `decoder_ML.md` §2), so reproduction means re-implementation from Methods. This one ships: Hydra configs, a train/inference runner, precomputed frames and DEM bundles, Stim `.dets` read/write with metadata, CUDA-Q QEC test-data generation, ONNX export, **INT8/FP8 quantisation and TensorRT inference**, a decoder-ablation workflow against several global decoders including `cudaq-qec` BP variants, and pretrained surface- and colour-code models on Hugging Face (separately licensed, access-controlled).

**Why the architecture is the right shape for us.** It is a **pre-decoder**, not a replacement decoder: a 3D CNN with a fixed local receptive field ($R=9$ to $R=17$) runs locally and in parallel over the detector tensor, removes most physical errors, and passes the residual syndrome to a conventional global decoder (PyMatching for surface codes, Chromobius for colour codes). That gives three properties matching decoders and monolithic neural decoders both lack:

- **The right baseline is already in the loop.** It is reported as outperforming *correlated* PyMatching up to $d=13$ — the §7.3 baseline, not the weak one.
- **Local and parallel ⇒ hardware-realisable.** A fixed receptive field is a bounded-fan-in, fixed-cost-per-round computation; this is the structural property that makes an FPGA tier conceivable at all. End-to-end ~1 µs/round at large $d$ on GB300 GPUs, below 1 µs with block-wise multi-GPU parallelism.
- **Quantisation is already in the repo.** INT8/FP8 is the named route to a power-feasible implementation and it does not have to be built from scratch.

One further component is directly relevant to `DEM_prior.md`: a separate architecture that **estimates decoding-graph weights from syndrome statistics alone**, with no explicit circuit-level noise model — reported as near uncorrelated PyMatching and in places above correlated PyMatching. That is Path B of `DEM_prior.md` done by a network, and it attacks the §5 "wrong or drifting $p_e$" row without replacing the decoder at all.

**Two companion papers from one group, both with repos, both directly on our axes.** [Yan, Li, Du, *Rethink the role of neural decoders in quantum error correction*, arXiv:2605.12046](https://arxiv.org/abs/2605.12046) (code: <https://github.com/GrahamYan/QEC-neural-decoder-benchmark>) is the one to read **first**: it unifies representative neural decoders into **five architectural paradigms**, benchmarks them to $d=9$ (161 qubits) under explicit accuracy *and* latency constraints, and runs an end-to-end compression pipeline onto **FPGA**. Three findings matter to us — near-term accuracy is driven more by **training-data volume than by architectural complexity**; an appropriate **inductive bias** is what buys accuracy; and **INT4** (not INT8) is required to reach microsecond latency on FPGA. The first finding is a warning about where effort goes; the third is a harder quantisation target than the NVIDIA repo's INT8/FP8 and should be treated as the real number for an FPGA tier.

**What that repo actually ships** (checked 2026-10-14; 6 commits, last in July 2026). It is about 1.8k lines of PyTorch:
- five model classes: MLP, dilated 3D CNN, TCN, transformer, and GNN/neural-BP;
- Stim data generation and training loops;
- Brevitas quantisation-aware and post-training quantisation at W4A4 and W8A8, with L1 pruning and batch-norm folding.

It does **not** ship:
- FPGA/HLS code or ONNX export;
- latency measurement;
- a matching baseline (PyMatching is not even a dependency);
- pretrained weights;
- a **licence**. Read it, but do not copy it into our open-source release.

The paper's FPGA half has to be rebuilt. Its experiments are also fixed-block, with `rounds = d` at uniform $p=5\times10^{-3}$, and the CNN takes $r$ as a constructor argument. None of the models streams round by round, so none fits a 25–100-round memory experiment as written.

[Yan, Li, Xiao, Ma, Cao, Pan, Du, *Efficient foundation decoders for fault-tolerant quantum computing*, arXiv:2606.27119](https://arxiv.org/abs/2606.27119) (code: <https://github.com/GrahamYan/ntu-decoder>) is the scaling counterpart: **neural transfer unification (NTU)** aligns decoding tasks across code distances using algebraic structure shared within a code family, so training on small codes amortises into large ones — the direct answer to "syndrome generation and training cost explode with $d$". The NTU-Transformer is reported to beat **correlation-aware matching** on the $[[361,1,19]]$ surface code, to exceed standard matching at $[[625,1,25]]$ through transfer, and to beat **Relay-BP** on the $[[72,12,6]]$ bivariate-bicycle code at low $p$. Note which baselines those are: correlation-aware matching (§6a) and Relay-BP (§5.1), not bare PyMatching — this is the §7.3 standard being met, and it is also the first entry here that engages §5.1's non-matchable codes.

Secondary options, for contrast rather than as a starting point: [Lange et al., PRResearch 7, 023181 (2025)](https://arxiv.org/abs/2307.01241) — a GNN that casts decoding as graph classification and beats matching under circuit noise *even when matching is given the exact error model*, though the authors released no code (a third-party partial reimplementation exists at <https://github.com/matteb29/gnn-surface-code-decoder>); and `decoder_ML.md` §2 for AlphaQubit/AQ2 as the accuracy and throughput reference points.

### 7.5 Where the opening is

Head-on accuracy competition with AQ2 is not winnable — $7.3\times10^{-11}$ per cycle at $d=23$, on DeepMind compute. Nor is the generic "small neural decoder on reconfigurable hardware" question still open; §7.4's two Yan et al. papers have taken the architecture survey, the FPGA deployment and the cross-distance scaling. What remains, and what the pre-decoder architecture is already shaped for:

1. **Leakage as a learned input, validated on hardware.** AlphaQubit saw leakage only through a simulated Pauli+ model; the $d=3$ experiment above *post-selected it away* at ~4.5 % per round. A decoder that ingests leakage flags and is tested against real leakage is an open, well-posed problem and the §5 leakage row is where it lands.
2. **Soft I/Q as a native channel** rather than a likelihood ratio bolted onto $w_e$.
3. **Streaming decoding over long runs, under an FPGA latency budget.** arXiv:2605.12046 benchmarks fixed-block decoding with $r=d$. It does not cover a decoder that:
   - updates its state (the Pauli frame) round by round over 25–100+ rounds;
   - beats MWPM on fitted per-round error $\varepsilon$;
   - still fits a per-round latency budget after INT4/INT8 quantisation.

   At $d=3$, neither axis alone is a claim. A lookup-table or union-find decoder already runs in nanoseconds on an FPGA. The simulated accuracy ceiling over MWPM is 16–21 % (`research_log/mwpm_headroom.md`). The claim is the combination. Treat that paper as the baseline to extend.

The evaluation metric for all three is a **Pareto point on (logical error per round $\varepsilon$, latency per round, FPGA resources)**, never accuracy alone. Fit $\varepsilon$ from $\mathrm{LER}(r)=\tfrac12\big(1-(1-2\varepsilon)^r\big)$ across several $r$. Per-shot LER at a single long $r$ saturates towards ½ for every decoder, including the optimal one.

### 7.6 Concrete first steps

Build the measurement apparatus before training anything; it is also what tells you whether the accuracy axis is worth pursuing.

1. **Baseline harness.**
   - Run Stim → PyMatching, correlated matching, belief-matching and gap-MWPM on identical shots: $d=3,5,7$, SI1000, $r=3,25,100$.
   - Report the fitted $\varepsilon$ per round (§7.5).
   - Reproduce §6's $1.15\times$ threshold and $7.8\times$ LER figures. If the baselines do not reproduce, nothing downstream is measurable.
   - Build this ourselves. `GrahamYan/QEC-neural-decoder-benchmark` has no matching baseline and no latency instrumentation (§7.4); its models and Brevitas code are a reference for the neural side, not a harness.
2. **Headroom measurement.**
   - Run Tesseract (§5.1) or small-$d$ exact MLD on the same shots. Exact MLD is feasible only at $d=3$, $r\le3$ (`research_log/mwpm_headroom.md`); at $r=25$, use Tesseract as a near-optimal proxy.
   - The spread between belief-matching and MLD *is* the maximum available accuracy gain.
   - If it is single-digit percent at affordable distance, the accuracy axis is the wrong headline. §7.5 point 3's latency framing then becomes the project, with accuracy reduced to "does not lose to correlated matching". Better learned in week two than month six.
3. **Reproduce a released decoder.**
   - Run `NVIDIA/Ising-Decoding` inference against the harness, and confirm the correlated-PyMatching comparison on your own Stim data before touching any architecture.
   - Then rebuild the arXiv:2605.12046 INT4 FPGA result as the latency reference point. The repo stops at Brevitas quantisation, so the export path has to be built: Brevitas → QONNX → FINN, or hls4ml.
   - That number, not AQ2's TPU throughput, is what our FPGA tier has to be argued against.
4. **Then** the ladder in `decoder_ML.md` §3.4, with its step-1 baseline upgraded from PyMatching to belief-matching.


---

## References

- Y. Kim, *Matching decoder — how it works?*, OSU group slides, 13 Sep 2026 — local copy: [`./QEC/MWPM_slides.pdf`](./MWPM_slides.pdf).
- O. Higgott, C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching*, Quantum 9, 1600 (2025), [arXiv:2303.15933](https://arxiv.org/abs/2303.15933).
- O. Higgott, T. C. Bohdanowicz, A. Kubica, S. T. Flammia, E. T. Campbell, *Improved decoding of circuit noise and fragile boundaries of tailored surface codes*, PRX 13, 031007 (2023), [arXiv:2203.04948](https://arxiv.org/abs/2203.04948).
- S. Bravyi, M. Suchara, A. Vargo, *Efficient algorithms for maximum likelihood decoding in the surface code*, PRA 90, 032326 (2014), [arXiv:1405.4883](https://arxiv.org/abs/1405.4883).
- B. Criger, I. Ashraf, *Multi-path summation for decoding 2D topological codes*, Quantum 2, 102 (2018), [arXiv:1709.02154](https://arxiv.org/abs/1709.02154).
- Z. Tobias, N. P. Breuckmann, B. Placke, *Optimal decoding with the worm*, [arXiv:2603.05428](https://arxiv.org/abs/2603.05428).
- L. A. Beni, O. Higgott, N. Shutty, *Tesseract: a search-based decoder for quantum error correction*, [arXiv:2503.10988](https://arxiv.org/abs/2503.10988); D. Grbic, L. A. Beni, N. Shutty, *Accelerating the Tesseract decoder*, [arXiv:2602.02985](https://arxiv.org/abs/2602.02985).
- C. Gidney, M. Newman, P. Brooks, C. Jones, *Yoked surface codes*, [arXiv:2312.04522](https://arxiv.org/abs/2312.04522).
- N. Meister, C. A. Pattison, J. Preskill, *Efficient soft-output decoders for the surface code*, [arXiv:2405.07433](https://arxiv.org/abs/2405.07433).
- A. Kubica, N. Delfosse, *Efficient color code decoders in $d\ge2$ dimensions from toric code decoders*, Quantum 7, 929 (2023), [arXiv:1905.07393](https://arxiv.org/abs/1905.07393).
- K. Sahay, B. J. Brown, *Decoder for the triangular color code by matching on a Möbius strip*, PRX Quantum 3, 010310 (2022), [arXiv:2108.11395](https://arxiv.org/abs/2108.11395); C. Gidney, C. Jones, *New circuits and an open source decoder for the color code*, [arXiv:2312.08813](https://arxiv.org/abs/2312.08813).
- P. Panteleev, G. Kalachev, *Degenerate quantum LDPC codes with good finite length performance*, Quantum 5, 585 (2021), [arXiv:1904.02703](https://arxiv.org/abs/1904.02703); J. Roffe et al., *Decoding across the quantum LDPC code landscape*, PRR 2, 043423 (2020), [arXiv:2005.07016](https://arxiv.org/abs/2005.07016).
- T. Hillmann et al., *Localized statistics decoding*, [arXiv:2406.18655](https://arxiv.org/abs/2406.18655); S. Wolanski, B. Barber, *Ambiguity clustering*, [arXiv:2406.14527](https://arxiv.org/abs/2406.14527); T. Müller et al., *Improved belief propagation is sufficient for real-time decoding of quantum memory* (relay-BP), [arXiv:2506.01779](https://arxiv.org/abs/2506.01779).
- M. Cain et al., *Correlated decoding of logical algorithms with transversal gates*, [arXiv:2403.03272](https://arxiv.org/abs/2403.03272).
- R. Blume-Kohout, K. Young, *Estimating detector error models from syndrome data*, [arXiv:2504.14643](https://arxiv.org/abs/2504.14643).
- O. Shtanko, A. Roy, A. Kumar, I. Minev, *Measure now, mitigate later: virtual error cancellation*, [arXiv:2610.12400](https://arxiv.org/abs/2610.12400) — Suppl. Props. 4–6 for the parity argument of §4.2 and the conditions (symplectic $Y$ weights, gauge generators, hook hyperedges) under which ties appear.
- *Decoder switching: breaking the speed–accuracy tradeoff in real-time QEC*, [arXiv:2510.25222](https://arxiv.org/abs/2510.25222).
- Google Quantum AI, *Suppressing quantum errors by scaling a surface code logical qubit*, Nature 614, 676 (2023), [arXiv:2207.06431](https://arxiv.org/abs/2207.06431) — $p_{ij}$ DEM calibration and correlated matching in practice.
- C. Chamberland, J. Olle, M. Li, S. Thornton, I. Baratta, *Fast and accurate AI-based pre-decoders for surface codes*, [arXiv:2604.12841](https://arxiv.org/abs/2604.12841) (2026) — open source: <https://github.com/NVIDIA/Ising-Decoding> (Apache-2.0). **The §7.4 starting point.**
- N. Sundaresan, T. J. Yoder, Y. Kim, M. Li, E. H. Chen, G. Harper, T. Thorbeck, A. W. Cross, A. D. Corcoles, M. Takita, *Demonstrating multi-round subsystem quantum error correction using matching and maximum likelihood decoders*, Nature Communications 14, 2852 (2023), [doi:10.1038/s41467-023-38247-5](https://doi.org/10.1038/s41467-023-38247-5) — the d=3 matching-vs-MLD comparison on identical shots quoted in §7.3.
- J. Bausch et al., *Learning high-accuracy error decoding for quantum processors* (AlphaQubit), Nature 635, 834 (2024); A. Senior, M. Edlich, N. Heras et al., *A scalable and real-time neural decoder for topological quantum codes* (AlphaQubit 2), [arXiv:2512.07737](https://arxiv.org/abs/2512.07737) (2026). No code or weights released; see `decoder_ML.md` §2.
- G. Yan, S. Li, Y. Du, *Rethink the role of neural decoders in quantum error correction*, [arXiv:2605.12046](https://arxiv.org/abs/2605.12046) (2026) — code: <https://github.com/GrahamYan/QEC-neural-decoder-benchmark>. Five architectural paradigms benchmarked to d=9 under joint accuracy/latency constraints, with an FPGA compression pipeline; data volume beats architectural complexity, inductive bias is decisive, INT4 is required for microsecond latency. **Read first (§7.4).** The repo ships models and Brevitas quantisation only: no FPGA code, latency harness, matching baseline or licence (§7.4).
- G. Yan, S. Li, S. Xiao, P. Ma, H. Cao, F. Pan, Y. Du, *Efficient foundation decoders for fault-tolerant quantum computing*, [arXiv:2606.27119](https://arxiv.org/abs/2606.27119) (2026) — code: <https://github.com/GrahamYan/ntu-decoder>. Neural transfer unification amortises training across code distances; NTU-Transformer beats correlation-aware matching at [[361,1,19]], standard matching at [[625,1,25]] via transfer, and Relay-BP at [[72,12,6]].
- M. Lange, P. Havstrom, B. Srivastava, I. Bengtsson, V. Bergentall, K. Hammar, O. Heuts, E. van Nieuwenburg, M. Granath, *Data-driven decoding of quantum error correcting codes using graph neural networks*, Phys. Rev. Research 7, 023181 (2025), [arXiv:2307.01241](https://arxiv.org/abs/2307.01241).
