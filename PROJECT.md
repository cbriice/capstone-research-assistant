# Machine Learning Decoders for Quantum Error Correction

**Project Sponsor:** Dr. Youngseok Kim, Electrical and Computer Engineering, The Ohio State University

## Project Description

Quantum computers cannot run useful algorithms without quantum error correction (QEC). In QEC, a logical qubit is encoded across many physical qubits, and repeated "stabilizer" measurements produce a stream of syndrome bits every microsecond. A classical program called a decoder must read that stream and infer which correction to apply. Decoding is a pure classical data-processing problem — large volumes of binary data with complex correlations in space and time — and it is now one of the main bottlenecks in building a fault-tolerant quantum computer.

Machine learning has become a leading approach: neural decoders built on recurrent networks, graph neural networks, transformers, and state-space models have matched or beaten hand-designed algorithms, including on real superconducting-processor data. This project uses a recent survey of the field (C. Lee, T. Hur, J. Jae, D. K. Park, "Machine Learning Approaches to Decoding Topological Quantum Codes," arXiv:2608.15760) as its roadmap: students will review the ML techniques used for QEC decoding, analyze the existing open-source code bases, and then implement and benchmark selected decoders themselves.

No quantum physics background is required. Students work in Python with standard ML tooling; an existing simulator generates the training data, so from the students' perspective this is a supervised learning problem on structured binary data under a hard real-time constraint.

## Objectives

- **Semester 1 — Survey, reproduce, and baseline.** Review the ML decoding literature by architecture family. Perform a codebase analysis of the major open-source decoder repositories (Stim, PyMatching, published neural decoders), documenting their structure, data formats, and design choices. Build a pipeline generating labeled syndrome datasets for the surface code under circuit-level noise, and reproduce a minimum-weight-perfect-matching (MWPM) decoder as the accuracy baseline.
- **Semester 2 — Implement, benchmark, and profile.** Implement two ML decoders of increasing sophistication (a feedforward/CNN decoder, then a recurrent or transformer-based one). Benchmark logical error rate against MWPM on simulated data and on the public Google surface-code experimental dataset, and profile inference latency against the ~1 microsecond-per-round budget real hardware imposes.
- **Stretch goal — Hardware prototyping.** Compress the best model (quantization, pruning) and deploy it to an FPGA or embedded accelerator, measuring real decoding latency in hardware rather than in a Python profiler.

## Needs and Requirements

- **Software:** Python, PyTorch, Stim (quantum circuit simulator), PyMatching. All open source and free.
- **Compute:** One workstation-class GPU suffices for the code sizes in scope; departmental or OSC resources can be arranged if needed.
- **Data:** Self-generated via simulation, plus the public Google Quantum AI surface-code syndrome dataset for validation against real hardware measurements.
- **Hardware (stretch goal only):** An FPGA board or embedded GPU module. No quantum or cryogenic hardware is needed — the project is entirely classical.
- **Mentoring:** Weekly meetings with the PI; QEC background provided in the first few weeks so students can start with no prior exposure.

## Challenges

- **Rare-event statistics:** logical errors are by design uncommon, so benchmarking requires large datasets and careful statistical treatment.
- **Degeneracy:** many distinct physical error patterns produce the identical syndrome, so the learning target must be the logical error class, not the physical error. Framing the label space correctly is a real conceptual hurdle.
- **Generalization:** a model trained at one code size or noise level must not collapse when either changes.
- **Accuracy versus latency:** the most accurate models are the slowest, and a decoder that cannot keep pace with the syndrome stream is useless however accurate. This tradeoff is the core of the project.

## Desired Skills

- **Required:** Python programming; comfort with linear algebra and basic probability; willingness to read research papers.
- **Strongly preferred (at least one team member):** Prior coursework or project experience in machine learning / deep learning, and familiarity with PyTorch.
- **Useful:** Software engineering practice (Git, testing, profiling); digital design and Verilog/HLS for the hardware stretch goal; any exposure to information or coding theory.
- **Not required:** Quantum mechanics or quantum computing background.

## Deliverables

A documented code-base analysis report; working, tested decoder implementations with reproducible benchmarks against classical baselines; an accuracy-versus-latency study across architectures; and, if the stretch goal is reached, a hardware demonstration of real-time inference. All software released as an open-source repository.
