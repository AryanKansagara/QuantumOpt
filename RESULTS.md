# Local results

Python 3.12, Qiskit 2.6.0, PennyLane 0.45.1, NumPy 2.5.4. Noiseless classical statevector simulation. Eight unit tests passed.

| Graph | Exact optimum | Greedy cut | Uniform expectation | QAOA expectation | Probability of exact-optimal outcome |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 nodes, 5 weighted edges | 4.1 | 4.1 | 2.3 | 3.24735 | 0.576874 |
| 6 nodes, 8 weighted edges | 6.9 | 6.9 | 3.45 | 4.78222 | 0.218942 |

Each search used 192 parameter pairs at p=1. At the selected parameters, cross-framework maximum absolute probability differences were below 1e-10. Drawing 4,096 simulated samples with seed 42 found the exact optimum on both instances.

The QAOA expected objectives improved on uniform sampling, but did not outperform the simple greedy reference on these graphs. Finding an optimal sample on two tiny instances does not demonstrate quantum advantage, speedup, hardware feasibility, or generalization. The comparison is educational and is not an exhaustive algorithm benchmark.
