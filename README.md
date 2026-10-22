# QuantumOpt: QAOA Optimization and Cross-Framework Validation

A Python project implementing p=1 QAOA for small weighted MaxCut instances in Qiskit and PennyLane. It compares exact noiseless distributions, a coarse parameter search, sampled outcomes, exhaustive optimum, and a simple classical greedy reference. Developed with AI assistance and verified with automated tests and cross-framework numerical checks.

## Run

Python 3.12 is used by the automated verification workflow. To reproduce:

```sh
git clone https://github.com/AryanKansagara/QuantumOpt.git
cd QuantumOpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python quantum_opt.py --output build/results.json
```

The included graphs have four and six nodes. Each parameter search evaluates 192 gamma/beta pairs. Qiskit Statevector and PennyLane default.qubit simulate the same H/RZZ/RX circuit. A bit-order conversion aligns qubit indexing before comparing full probability vectors.

See [RESULTS.md](RESULTS.md) for the measured local results. On Windows, activate the environment with `.venv\Scripts\activate` instead of the `source` command.

## Mathematical contract

For positive edge weights, C(z)=sum w_uv * [z_u != z_v]. Initialize |+> on every wire. The cost unitary exp(-i gamma C) is implemented by RZZ(-gamma*w_uv), ignoring a global phase; the mixer uses RX(2*beta). Maximize the expectation of C. The exact reference enumerates all 2^n assignments. A simulated sample's best cut is distinct from the circuit's expected cut; neither establishes superiority over a classical optimizer.

## Evaluation and limitations

The report records expected cut, exact optimum, expected-cut/optimum ratio, optimal-outcome probability, best cut in 4,096 simulated draws, and cross-framework agreement. Tests cover known objectives, invalid graph data/angles, normalization, bit-order consistency, objective bounds, and improvement over uniform sampling on one toy graph.

This is algorithm implementation and testing exposure, not quantum-computing expertise, a novel algorithm, quantum hardware deployment, noise robustness, or quantum advantage. The simulator and exhaustive baseline both grow exponentially in qubit count. The finite grid need not find globally optimal circuit parameters. MaxCut is a graph optimization demonstration, not financial instrument valuation, portfolio optimization, or a trained quantum machine-learning model.

## Primary references

- [PennyLane QAOA for MaxCut tutorial](https://pennylane.ai/demos/tutorial_qaoa_maxcut)
- [Qiskit Statevector API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info.Statevector)

The implementation uses a negative RZZ cost angle to represent exp(-i gamma C) directly; sign conventions can be absorbed into the variational parameter range.
