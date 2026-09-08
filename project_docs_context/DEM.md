# Detector Error Models — A Summary

*Author: Youngseok Kim*
*Last updated: 2026-04-29*

This note summarizes the Detector Error Model (DEM) framework as developed in Blume-Kohout & Young, *Estimating detector error models from syndrome data* [^bk2025]. It covers what a DEM is, how it relates to syndrome measurement, how it relates to logical error, and how it differs mathematically from a conventional Pauli-channel description.

---

## 1. What is a Detector Error Model?

A DEM is a **statistical model for the patterns of detector-bit flips produced by a noisy syndrome-extraction circuit**. Concretely it is a finite list

$$\mathrm{DEM} = \{(s_i, p_{s_i}) : i = 1, \ldots, L\},$$

where each $s_i \in \{0,1\}^N$ is the *support* of a single DEM event — a fixed subset of the $N$ detector bits over the entire experiment that this event flips together — and $p_{s_i} \in [0,1]$ is the probability that this event fires *independently* of all other events.

The defining rules of the model are deliberately phenomenological [^bk2025, §1.2]:

1. **Independence.** Events fire as independent Bernoulli trials.
2. **Support-defined identity.** A DEM event is identified entirely by which detectors it flips.
3. **Indistinguishables are merged.** Two distinct physical mechanisms that always flip *exactly* the same detector subset are treated as a single DEM event with combined probability — the DEM never tries to separate things the data can't separate.

The sampling rule is correspondingly simple: draw an $L$-vector $\mathbf{c}$ with $c_i \stackrel{\text{ind}}{\sim} \mathrm{Bernoulli}(p_{s_i})$, then the detector history is

$$x \;=\; \bigoplus_{i:\,c_i = 1} s_i \;\in\; \{0,1\}^N.$$

A DEM therefore defines a distribution $\Pr_{\mathrm{DEM}}$ on $\{0,1\}^N$, which is the object the decoder actually consumes.

### 1.1 Polarizations and attenuations

It is convenient to work in the *polarization* basis (Walsh–Hadamard transform of the probability vector):

$$z_y \equiv (-1)^{x \cdot y},\qquad \langle z_y\rangle \;=\; \prod_{i:\,s_i\cdot y\,=\,1} (1 - 2 p_{s_i}),$$

where $y \in \{0,1\}^N$ indexes a parity to evaluate, and $s_i \cdot y$ is the GF(2) inner product. Defining the **attenuation** of event $i$,

$$a_{s_i} \;\equiv\; -\ln(1 - 2 p_{s_i}) \;\approx\; 2 p_{s_i} \quad (\text{for small } p_{s_i}),$$

and the **depolarization** $\omega_y \equiv -\ln\langle z_y\rangle$, the polarization formula becomes linear:

$$\boxed{\;\omega_y \;=\; \sum_{i:\,s_i\cdot y\,=\,1} a_{s_i}\;}\qquad (\star)$$

This linearity in the attenuations is the central algebraic feature of DEMs and the reason they are cleanly estimable from data [^bk2025, §2.2].

---

## 2. Relation to Syndrome Measurement

The chain that connects raw measurements to DEM events is

$$\underbrace{\text{syndromes}}_{(M+1)\times(n-k)\text{ bits}} \;\xrightarrow{\;\mathrm{XOR}\;}\; \underbrace{\text{detectors}}_{N = M(n-k)\text{ bits}} \;\xrightarrow{\;\mathrm{DEM}\;}\; \underbrace{\text{events}}_{\{(s_i, p_{s_i})\}}.$$

- A **syndrome history** is what is actually measured: an $M \times (n-k)$ array of bits, one row per syndrome-extraction cycle (SEC), one column per stabilizer ancilla.
- A **detector** is the XOR of two consecutive measurements of the same syndrome bit. If a stabilizer's value does not change between cycle $t$ and $t+1$, that detector bit reads 0; it flips to 1 only when *something has changed*. With an initial $t=0$ all-zero baseline plus $M$ measured cycles, this gives an $M \times (n-k)$ detector array, flattened into the $N$-bit string above [^bk2025, §1.2].
- A DEM event is exactly the **deterministic detector signature** of a single physical error mechanism. For example, in the paper's two-cycle $d=3$ repetition-code worked example:
  - A data-qubit X flip $Q_1(t_{0.5})$ between $t=0$ and $t=1$ flips a single detector → support $\{1\}$.
  - A readout error $S_1(t_1)$ on ancilla 1 at cycle $t=1$ flips *two* detectors at non-adjacent times → support $\{1,3\}$ (one when the wrong reading appears, one when it "changes back").
  - In the same example, $Q_1(t_{1.5})$ and $S_1(t_2)$ have *identical* support $\{3\}$ — they are physically distinct but indistinguishable in the detector record, so the DEM lumps them into one event with combined rate.

Reading off this chain, the DEM is the dictionary that turns *patterns observed in the XOR-of-syndromes* into *probabilities of underlying error events*. Syndromes are both the input data the DEM is built from and the input data the DEM is applied to at inference time.

---

## 3. Relation to Logical Error

Logical errors enter through the **decoder**, and the DEM is what the decoder uses as its prior. The flow is:

1. **Measure** $M$ cycles of syndromes → compute detector history $x \in \{0,1\}^N$.
2. **Decode**: given the DEM and the observed $x$, find the most-likely set $\{c_i\}$ of DEM events whose XOR equals $x$:
   $$\hat{\mathbf{c}}(x) \;=\; \arg\max_{\mathbf{c}:\;\bigoplus_i c_i s_i = x}\; \prod_i p_{s_i}^{c_i}(1 - p_{s_i})^{1-c_i}.$$
3. **Predict logical outcome**: in a full DEM each event carries an indicator of whether it flips a given logical observable; the decoder's $\hat{\mathbf{c}}(x)$ therefore induces a guess of the logical-frame correction.
4. **Logical error** occurs when the decoder's guess of the logical frame disagrees with the truth — that is, when the most-likely-under-DEM explanation of $x$ and the true realization differ on at least one logical observable.

Two structural failure modes follow:

- **Degeneracy.** The map "event set → detector pattern" is many-to-one. The decoder picks the most probable explanation under the DEM's $\{p_{s_i}\}$; if reality picked a less-probable one, you lose. *Better DEM estimates → lower logical error rate*, because a more accurate prior makes the decoder guess wrong less often.
- **Undetectable events.** Any DEM event whose support is empty (or whose effect cancels on detectors but acts non-trivially on the logical observable) is invisible by construction and propagates directly into a logical error. These set the irreducible floor on logical performance.

Quantitatively, if the estimated attenuations $\hat a_s$ differ from the truth $a_s$ by $\delta a_s = \hat a_s - a_s$, then by ($\star$) the decoder's log-odds for each candidate explanation shift by sums of these $\delta a_s$. Near a small-error operating point this propagates approximately linearly into the logical error rate — which makes precise estimation of the DEM directly equivalent to suppressing logical error.

---

## 4. Pauli Noise Model vs DEM

### 4.1 Pauli noise model

A Pauli model lives at the **physical Hilbert-space level**. For $n$ physical qubits it asserts that the noisy circuit implements a Pauli channel

$$\mathcal{E}(\rho) \;=\; \sum_{P \in \mathcal{P}_n}\, q_P\,P\rho P^\dagger,\qquad \sum_P q_P = 1,\quad q_P \ge 0,$$

with $4^n$ free parameters globally, or — under a factorized assumption with one local channel per circuit location $\ell$ — $\sum_\ell 4^{w_\ell}$ parameters, where $w_\ell$ is the location's weight.

### 4.2 DEM as Pauli mod stabilizers

Every Pauli error $P$ has a deterministic detector signature under the syndrome circuit; call this the **signature map**

$$\sigma : \mathcal{P}_n \;\longrightarrow\; \{0,1\}^N,\qquad P \mapsto s(P).$$

The DEM rate for support $s$ is the aggregated rate of all Paulis with that signature,

$$p_s \;=\; \mathrm{Pr}\Big[\bigoplus_{P:\,\sigma(P) = s} \mathbf{1}_{P\text{ occurred}} = 1\Big],$$

so the Pauli → DEM mapping is a **quotient by the kernel of $\sigma$**. The kernel always contains:

- the stabilizer group $\mathcal{S}$ (commutes with every stabilizer ⇒ flips no syndrome bits, $\sigma(\mathcal{S}) = 0$);
- the logical-operator coset $N(\mathcal{S})/\mathcal{S}$ (non-trivial on the code, but invisible to detectors).

Hence the paper's one-line characterization [^bk2025, §1.2]:

> *"Detector error models are essentially Pauli channels mod stabilizers — which is exactly what can be estimated uniquely from data."*

### 4.3 Side-by-side comparison

| | Pauli model | DEM |
|---|---|---|
| **What it describes** | Physical channel on $n$ qubits | Observable distribution on $N$ detector bits |
| **Parameter count** | $\sim 4^n$ globally, $\sum_\ell 4^{w_\ell}$ for factorized | $L$ = number of distinct detector signatures (typically polynomial in $d$ and circuit depth) |
| **Identifiable from syndromes alone?** | **No** — stabilizer / logical sub-channels are in $\ker(\sigma)$ and invisible | **Yes** — $(\star)$ gives an invertible linear map from $\{\omega_y\}$ to $\{a_{s_i}\}$ under standard conditions |
| **Underlying algebra** | Non-Abelian group $\mathcal{P}_n$ acting on a $2^n$-dim Hilbert space | Abelian group $(\mathbb{Z}/2)^N$ acting on a $2^N$-element sample space, Walsh-Hadamard diagonalizable |
| **Estimator from data** | Requires extra structure (cycle benchmarking, sparsity priors, etc.) | Direct: $\hat{\langle z_y\rangle} = \tfrac{1}{K}\sum_k (-1)^{x_k \cdot y}$, then $\hat\omega_y = -\ln\hat{\langle z_y\rangle}$, then solve $M \vec a = \hat{\vec\omega}$ |
| **Sufficient for decoding?** | Yes, but only via the induced DEM | Yes — the decoder consumes the DEM by definition |

### 4.4 Practical implications

- **Mis-estimating Pauli rates in $\ker(\sigma)$** has *zero effect* on decoder behaviour. From a QEC standpoint, any modelling effort spent on those degrees of freedom is wasted.
- **Mis-estimating DEM rates $p_s$** *does* shift the decoder's prior and propagates directly into the logical error rate via the linear log-odds shift implied by $(\star)$.
- The Walsh-Hadamard / polarization viewpoint also turns DEM estimation into the kind of generalized-Fourier inference problem that is well-understood from Pauli-channel learning [^bk2025, §2], so existing technical machinery transfers cleanly.

---

## 5. A Worked Example: Distance-3 Repetition Code with $M=2$ Cycles

This is the example the paper uses on page 3, traced explicitly so you can see how a Pauli inventory collapses into a DEM and why some Pauli rates are unlearnable from syndromes alone.

### 5.1 Setup

- **Code.** 3 data qubits $Q_1, Q_2, Q_3$, two $Z$-stabilizers $Z_1 Z_2$ and $Z_2 Z_3$, read by ancillas $S_1$ and $S_2$ respectively.
- **Cycles.** Measure $S_1, S_2$ at $t_1$, then again at $t_2$. Initial values at $t_0$ are 0.
- **Detectors.** $N = M(n-k) = 2\cdot 2 = 4$ detector bits formed by XOR-ing consecutive syndrome measurements:
  - $x_1 = S_1(t_0)\oplus S_1(t_1) = S_1(t_1)$
  - $x_2 = S_2(t_0)\oplus S_2(t_1) = S_2(t_1)$
  - $x_3 = S_1(t_1)\oplus S_1(t_2)$
  - $x_4 = S_2(t_1)\oplus S_2(t_2)$

### 5.2 Physical errors → detector signatures

Walk through each candidate single-Pauli error and figure out which syndrome bits it changes, then which detector bits flip. A data-qubit X error is *persistent* — it shifts every subsequent syndrome reading — while a readout error is *transient* and affects only one cycle:

| Physical error | What syndrome bits change | Detector pattern $[x_1\,x_2\,x_3\,x_4]$ | DEM support $s$ |
|---|---|---|---|
| $Q_1(t_{0.5})$ — X on data 1 before round 1 | $S_1$ flips at $t_1$, persists at $t_2$ | $[1\,0\,0\,0]$ | $\{1\}$ |
| $Q_2(t_{0.5})$ — X on data 2 before round 1 | both $S_1, S_2$ flip at $t_1$, persist | $[1\,1\,0\,0]$ | $\{1,2\}$ |
| $Q_3(t_{0.5})$ — X on data 3 before round 1 | only $S_2$ flips at $t_1$, persists | $[0\,1\,0\,0]$ | $\{2\}$ |
| $S_1(t_1)$ — readout 1 misread at $t_1$ | $S_1^{\text{read}}(t_1) = 1$, $S_1^{\text{read}}(t_2) = 0$ | $[1\,0\,1\,0]$ | $\{1,3\}$ |
| $S_2(t_1)$ — readout 2 misread at $t_1$ | analogous on $S_2$ | $[0\,1\,0\,1]$ | $\{2,4\}$ |
| $Q_1(t_{1.5})$ — X on data 1 between rounds | $S_1$ flips at $t_2$ only | $[0\,0\,1\,0]$ | $\{3\}$ |
| $Q_2(t_{1.5})$ — X on data 2 between rounds | $S_1, S_2$ both change at $t_2$ | $[0\,0\,1\,1]$ | $\{3,4\}$ |
| $S_1(t_2)$ — readout 1 misread at $t_2$ | $S_1^{\text{read}}(t_2) = 1$, true was 0 | $[0\,0\,1\,0]$ | $\{3\}$ |

Eight physically distinct errors collapse onto **seven** distinct detector signatures. The collision is

$$\sigma\big(Q_1(t_{1.5})\big) \;=\; \sigma\big(S_1(t_2)\big) \;=\; \{3\}.$$

(An analogous collision $\sigma(Q_3(t_{1.5})) = \sigma(S_2(t_2)) = \{4\}$ also occurs.) The DEM merges each colliding pair into one event.

### 5.3 Pauli view vs DEM view of the aggregate statistics

Suppose many shots produce $x = [0\,0\,1\,0]$ at frequency $f_{\{3\}}$.

**Pauli view.** You want two separate rates. To leading order,

$$f_{\{3\}} \;\approx\; p_{Q_1(t_{1.5})} \;+\; p_{S_1(t_2)}.$$

This is **one equation in two unknowns**. Syndrome data alone cannot separate them — the difference $p_{Q_1(t_{1.5})} - p_{S_1(t_2)}$ lives in $\ker(\sigma)$ and is invisible. To pin them down you need outside information (cycle benchmarking, an independent readout-error characterization, etc.).

**DEM view.** Define one event $s = [0010]$ with rate
$$p_{[0010]} \;=\; p_{Q_1(t_{1.5})}\,(1-p_{S_1(t_2)}) \;+\; (1-p_{Q_1(t_{1.5})})\,p_{S_1(t_2)},$$
or in attenuations,
$$a_{[0010]} \;=\; a_{Q_1(t_{1.5})} \;+\; a_{S_1(t_2)}.$$

This single number is identifiable: in the linear system $\omega_y = \sum_{s:\,s\cdot y=1} a_s$, observe the depolarization $\omega_{[0010]} \approx -\ln(1 - 2 f_{\{3\}})$, solve, and you have $a_{[0010]}$.

### 5.4 Inverse problem on a single shot

Now consider a single observed detector history $x = [1\,0\,1\,0]$. Which events fired? At least two consistent explanations from the table:

- **One event:** $S_1(t_1)$ alone, prior $p_{[1010]}$.
- **Two events:** $Q_1(t_{0.5})$ and $Q_1(t_{1.5})$ both fired, since $[1000]\oplus[0010] = [1010]$. Prior $\approx p_{[1000]}\,p_{[0010]}$.

The decoder picks the more likely explanation. For uniformly small $p$, the single-event explanation wins ($O(p)$ vs $O(p^2)$). Crucially, the decoder needs only the DEM probabilities $\{p_{[1000]}, p_{[0010]}, p_{[1010]}, \ldots\}$, not the underlying Pauli rates.

### 5.5 Why this affects logical error

Return to the $\{3\}$ collision and ask what the decoder does. It always applies the same correction to $x = [0\,0\,1\,0]$ — say, "$X_1$ on data qubit 1" — because both explanations are consistent with the syndrome:

- **True firing was $Q_1(t_{1.5})$.** Data qubit really was flipped. Decoder's $X_1$ correction restores the state. ✓
- **True firing was $S_1(t_2)$.** No data-qubit corruption — just a readout misreport. Decoder's $X_1$ correction *introduces* an actual $X_1$ error on the data. ✗ → logical error.

So whether $x=[0\,0\,1\,0]$ ends up as a logical error depends on which physical mechanism really fired, and the syndrome decoder cannot tell. The logical-error rate from this signature is determined entirely by the **ratio** $p_{Q_1(t_{1.5})} : p_{S_1(t_2)}$ — but for the decoder to do the right thing on average, all it needs is the aggregate rate and the relative weights, which is exactly what the DEM tracks.

This is the core distinction in three lines:

- The **Pauli model** wants to estimate $p_{Q_1(t_{1.5})}$ and $p_{S_1(t_2)}$ separately, but cannot from syndrome data alone.
- The **DEM** estimates the aggregate $p_{[0010]}$, which is identifiable and is exactly what the decoder needs.
- Any structure in the Pauli model **beyond** what the DEM captures is **invisible to the decoder** and **does not affect the logical error rate**.

*(Aside: the paper restricts attention to the syndrome history and omits the terminating logical measurement. If you include it, the **full** DEM tags each event with whether it flips the logical observable, so $Q_1(t_{1.5})$ and $S_1(t_2)$ would in fact split back into two distinct events — because they have opposite effects on the logical observable even though their syndrome signatures coincide. The simplified DEM in this section is the natural object for a decoder that consumes only the syndrome stream.)*

---

## 6. Takeaways

- A **DEM** is an Abelian, observable-level model: a list of (detector-support, probability) pairs whose Walsh–Hadamard transform expresses the data as a linear function of attenuations.
- **Syndrome measurements** are the source of detector bits, and each physical error has a deterministic detector signature; the DEM is the dictionary from detector patterns back to event probabilities.
- **Logical error** is what happens when the decoder, using the DEM as its prior, picks the wrong explanation of the observed detector pattern — so *more accurate DEMs directly buy lower logical error rates*.
- Compared to a **Pauli noise model**, a DEM parametrizes exactly the quotient $\mathcal{P}_n / \ker(\sigma)$ — i.e., the *learnable* part of the noise — discarding the stabilizer and logical components that syndrome data cannot resolve. This makes DEMs both **identifiable** from syndrome data alone and **sufficient** for predicting logical error rates.

---

## References

[^bk2025]: R. Blume-Kohout, K. Young, *Estimating detector error models from syndrome data*, Sandia National Laboratories; arXiv:2504.14643 (April 2025). <https://arxiv.org/abs/2504.14643>. (Source for: DEM definition (§1.2), polarization / attenuation formalism (§2.1–§2.2), "DEM = Pauli channels mod stabilizers", and the worked $d=3$ repetition-code example.)

[^spitz2018]: S. Spitz, B. Tarasinski, C. W. J. Beenakker, T. E. O'Brien, *Adaptive weight estimator for quantum error correction in a time-dependent environment*, Adv. Quantum Technol. (2018). (Original $p_{ij}$ moments-based DEM estimation.)

[^chen2021]: Z. Chen et al. (Google Quantum AI), *Exponential suppression of bit or phase errors with cyclic error correction*, Nature **595**, 383 (2021). (Empirical use of detector-bit moments.)
