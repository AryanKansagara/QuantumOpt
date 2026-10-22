"""Small-graph QAOA with Qiskit/PennyLane agreement and classical references."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def validate_graph(n, edges):
    if not isinstance(n, int) or not 2 <= n <= 8 or not edges:
        raise ValueError('Use 2–8 nodes and a nonempty edge list for this local demo')
    seen = set()
    for u, v, weight in edges:
        if not isinstance(u, int) or not isinstance(v, int) or not 0 <= u < v < n:
            raise ValueError('Edges must satisfy 0 <= u < v < n')
        if (u, v) in seen or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Weights must be positive/finite; duplicate edges forbidden')
        seen.add((u, v))


def cut_values(n, edges):
    validate_graph(n, edges)
    return np.array([sum(w for u, v, w in edges if ((bits >> u) & 1) != ((bits >> v) & 1))
                     for bits in range(2**n)], dtype=float)


def qiskit_probabilities(n, edges, gamma, beta):
    validate_graph(n, edges)
    if not math.isfinite(gamma) or not math.isfinite(beta):
        raise ValueError('Angles must be finite')
    circuit = QuantumCircuit(n)
    for wire in range(n):
        circuit.h(wire)
    # C = sum w*(I-ZuZv)/2. exp(-i gamma C) equals RZZ(-gamma*w), up to global phase.
    for u, v, weight in edges:
        circuit.rzz(-gamma*weight, u, v)
    for wire in range(n):
        circuit.rx(2*beta, wire)
    return np.asarray(Statevector.from_instruction(circuit).probabilities())


def pennylane_probabilities(n, edges, gamma, beta):
    validate_graph(n, edges)
    if not math.isfinite(gamma) or not math.isfinite(beta):
        raise ValueError('Angles must be finite')
    device = qml.device('default.qubit', wires=n)

    @qml.qnode(device)
    def circuit():
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for u, v, weight in edges:
            qml.IsingZZ(-gamma*weight, wires=[u, v])
        for wire in range(n):
            qml.RX(2*beta, wires=wire)
        return qml.probs(wires=range(n))

    native = np.asarray(circuit())
    # PennyLane wire 0 is the most significant bit; Qiskit qubit 0 is least significant.
    return np.array([native[int(format(i, f'0{n}b')[::-1], 2)] for i in range(2**n)])


def greedy_reference(n, edges):
    values = cut_values(n, edges)
    current = 0
    while True:
        candidates = [current ^ (1 << wire) for wire in range(n)]
        best = max(candidates, key=lambda bits: values[bits])
        if values[best] <= values[current]:
            return float(values[current])
        current = best


def benchmark(n, edges):
    values = cut_values(n, edges)
    optimum = float(values.max())
    best = (-1.0, None, None, None)
    for gamma in np.linspace(0, 2*np.pi, 16, endpoint=False):
        for beta in np.linspace(0, np.pi, 12, endpoint=False):
            probabilities = qiskit_probabilities(n, edges, float(gamma), float(beta))
            expectation = float(probabilities @ values)
            if expectation > best[0]:
                best = (expectation, float(gamma), float(beta), probabilities)
    expectation, gamma, beta, probabilities = best
    other = pennylane_probabilities(n, edges, gamma, beta)
    difference = float(np.max(np.abs(probabilities-other)))
    if difference > 1e-10:
        raise RuntimeError('Framework probability mismatch')
    samples = np.random.default_rng(42).choice(2**n, size=4096, p=probabilities/probabilities.sum())
    mask = np.isclose(values, optimum, rtol=0, atol=1e-12)
    return {'nodes': n, 'edges': len(edges), 'p': 1, 'grid_candidates': 192,
            'classical_exact_optimum': optimum, 'uniform_expected_cut': float(values.mean()),
            'classical_greedy_cut': greedy_reference(n, edges), 'qaoa_expected_cut': expectation,
            'expected_cut_over_optimum': expectation/optimum,
            'optimal_bitstring_probability': float(probabilities[mask].sum()),
            'sampled_best_cut_4096_shots': float(values[samples].max()),
            'cross_framework_max_probability_difference': difference,
            'best_gamma': gamma, 'best_beta': beta}


GRAPHS = [
    (4, [(0,1,1.0),(1,2,1.2),(2,3,0.8),(0,3,1.1),(0,2,0.5)]),
    (6, [(0,1,1.0),(1,2,0.7),(2,3,1.1),(3,4,0.8),(4,5,1.3),(0,5,0.9),(0,3,0.5),(1,4,0.6)]),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'method': 'Noiseless classical statevector simulation; no quantum advantage claimed',
              'benchmarks': [benchmark(n, edges) for n, edges in GRAPHS]}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)
