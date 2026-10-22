import unittest
import numpy as np
from quantum_opt import GRAPHS, benchmark, cut_values, pennylane_probabilities, qiskit_probabilities, validate_graph


class QuantumTests(unittest.TestCase):
    def test_known_single_edge(self):
        self.assertEqual(cut_values(2, [(0,1,1)]).tolist(), [0,1,1,0])

    def test_known_square_optimum(self):
        self.assertEqual(cut_values(4, [(0,1,1),(1,2,1),(2,3,1),(0,3,1)]).max(), 4)

    def test_bad_graph(self):
        for n, edges in [(9, [(0,1,1)]), (2, [(0,0,1)]), (2, [(0,1,-1)]), (2, [(0,1,1),(0,1,1)])]:
            with self.subTest(n=n, edges=edges), self.assertRaises(ValueError):
                validate_graph(n, edges)

    def test_uniform_start(self):
        n, edges = GRAPHS[0]
        np.testing.assert_allclose(qiskit_probabilities(n,edges,0,0), np.full(2**n,1/2**n), atol=1e-12)

    def test_cross_framework_distributions(self):
        for n, edges in GRAPHS:
            for gamma, beta in [(0,0),(0.3,0.7),(1.1,0.2),(2.0,0.4)]:
                with self.subTest(n=n,gamma=gamma,beta=beta):
                    a=qiskit_probabilities(n,edges,gamma,beta)
                    b=pennylane_probabilities(n,edges,gamma,beta)
                    self.assertAlmostEqual(float(a.sum()),1)
                    np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)

    def test_expected_objective_bounds(self):
        n, edges = GRAPHS[0]
        costs=cut_values(n,edges)
        expectation=float(qiskit_probabilities(n,edges,0.5,0.4)@costs)
        self.assertGreaterEqual(expectation,0)
        self.assertLessEqual(expectation,costs.max()+1e-12)

    def test_benchmark_improves_uniform_on_toy(self):
        result=benchmark(*GRAPHS[0])
        self.assertGreater(result['qaoa_expected_cut'],result['uniform_expected_cut'])
        self.assertLessEqual(result['sampled_best_cut_4096_shots'],result['classical_exact_optimum'])
        self.assertLess(result['cross_framework_max_probability_difference'],1e-10)

    def test_invalid_angles(self):
        with self.assertRaises(ValueError):
            qiskit_probabilities(*GRAPHS[0],float('nan'),0)


if __name__ == '__main__':
    unittest.main()
