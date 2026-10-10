"""
Tests for evaluation metrics, confidence validation, alignment error,
information loss, and Bayesian optimisation.
"""

import unittest
import math
from context_memory.memory_store import Memory
from context_memory.retrieval import DenseRetriever, TfidfVectoriser
from context_memory.evaluation import (
    retrieval_precision, retrieval_recall, f1_score,
    redundancy_rate, token_budget_utilisation, selection_quality_report,
)
from context_memory.confidence_validation import (
    evidence_density, agreement_score, coverage_score, composite_confidence,
)
from context_memory.memory_alignment import (
    measure_token_dependence, alignment_error, full_alignment_report,
)
from context_memory.information_loss import (
    token_information_loss, coverage_loss, full_information_loss_report,
)
from context_memory.bayesian_optimization import (
    BayesianOptimiser, rbf_kernel, gp_predict, expected_improvement,
)


class TestPrecisionRecall(unittest.TestCase):
    def test_perfect_precision(self):
        p = retrieval_precision(["a", "b"], {"a", "b", "c"})
        self.assertAlmostEqual(p, 1.0)

    def test_zero_precision(self):
        p = retrieval_precision(["x", "y"], {"a", "b"})
        self.assertAlmostEqual(p, 0.0)

    def test_partial_precision(self):
        p = retrieval_precision(["a", "x"], {"a", "b"})
        self.assertAlmostEqual(p, 0.5)

    def test_recall(self):
        r = retrieval_recall(["a"], {"a", "b"})
        self.assertAlmostEqual(r, 0.5)

    def test_f1(self):
        self.assertAlmostEqual(f1_score(1.0, 1.0), 1.0)
        self.assertAlmostEqual(f1_score(0.0, 0.0), 0.0)
        self.assertAlmostEqual(f1_score(0.5, 0.5), 0.5)

    def test_empty_retrieved(self):
        self.assertAlmostEqual(retrieval_precision([], {"a"}), 0.0)

    def test_empty_relevant(self):
        self.assertAlmostEqual(retrieval_recall(["a"], set()), 0.0)


class TestRedundancyRate(unittest.TestCase):
    def test_no_redundancy_with_one_memory(self):
        m = Memory(id="a", text="test", embedding=[1.0, 0.0])
        self.assertAlmostEqual(redundancy_rate([m]), 0.0)

    def test_high_redundancy_identical_embeddings(self):
        m1 = Memory(id="a", text="test1", embedding=[1.0, 0.0, 0.0])
        m2 = Memory(id="b", text="test2", embedding=[1.0, 0.0, 0.0])
        self.assertAlmostEqual(redundancy_rate([m1, m2], threshold=0.9), 1.0)


class TestTokenBudget(unittest.TestCase):
    def test_within_budget(self):
        mems = [Memory(id="a", text="short")]
        result = token_budget_utilisation(mems, 100)
        self.assertTrue(result["within_budget"])

    def test_utilisation(self):
        m = Memory(id="a", text="one two three four five")
        result = token_budget_utilisation([m], 100)
        self.assertGreater(result["utilisation"], 0)
        self.assertLessEqual(result["utilisation"], 1.0)


class TestEvidenceDensity(unittest.TestCase):
    def test_full_coverage(self):
        query = "python flask"
        memories = [Memory(id="a", text="python flask web development")]
        ed = evidence_density(query, memories)
        self.assertAlmostEqual(ed, 1.0)

    def test_no_coverage(self):
        query = "quantum physics"
        memories = [Memory(id="a", text="cooking recipe pasta")]
        ed = evidence_density(query, memories)
        self.assertAlmostEqual(ed, 0.0)

    def test_partial_coverage(self):
        query = "python java rust"
        memories = [Memory(id="a", text="python programming language")]
        ed = evidence_density(query, memories)
        self.assertGreater(ed, 0.0)
        self.assertLess(ed, 1.0)

    def test_empty_query(self):
        ed = evidence_density("", [Memory(id="a", text="test")])
        self.assertAlmostEqual(ed, 0.0)


class TestAlignmentError(unittest.TestCase):
    def test_perfect_alignment(self):
        err = alignment_error(0.5, 0.5)
        self.assertAlmostEqual(err, 0.0)

    def test_maximum_misalignment(self):
        err = alignment_error(0.0, 1.0)
        self.assertAlmostEqual(err, 1.0)

    def test_partial_misalignment(self):
        err = alignment_error(0.7, 0.3)
        self.assertAlmostEqual(err, 0.4)

    def test_invalid_preference_raises(self):
        with self.assertRaises(ValueError):
            alignment_error(1.5, 0.5)

    def test_invalid_dependence_raises(self):
        with self.assertRaises(ValueError):
            alignment_error(0.5, -0.1)


class TestTokenDependence(unittest.TestCase):
    def test_high_dependence(self):
        response = "python flask web backend development"
        memories = [Memory(id="a", text="python flask web backend development")]
        dep = measure_token_dependence(response, memories)
        self.assertAlmostEqual(dep, 1.0)

    def test_low_dependence(self):
        response = "quantum physics electrons"
        memories = [Memory(id="a", text="cooking pasta recipe")]
        dep = measure_token_dependence(response, memories)
        self.assertAlmostEqual(dep, 0.0)

    def test_empty_response(self):
        dep = measure_token_dependence("", [Memory(id="a", text="test")])
        self.assertAlmostEqual(dep, 0.0)


class TestTokenInformationLoss(unittest.TestCase):
    def test_no_loss(self):
        originals = ["python flask web"]
        retained = ["python flask web backend"]
        loss = token_information_loss(originals, retained)
        self.assertAlmostEqual(loss, 0.0)

    def test_total_loss(self):
        originals = ["python flask"]
        retained = ["java spring"]
        loss = token_information_loss(originals, retained)
        self.assertAlmostEqual(loss, 1.0)

    def test_partial_loss(self):
        originals = ["python flask web"]
        retained = ["python java"]
        loss = token_information_loss(originals, retained)
        self.assertGreater(loss, 0.0)
        self.assertLess(loss, 1.0)


class TestBayesianOptimisation(unittest.TestCase):
    def test_rbf_kernel_identical(self):
        x = [1.0, 2.0]
        k = rbf_kernel(x, x)
        self.assertAlmostEqual(k, 1.0)

    def test_rbf_kernel_distant(self):
        a = [0.0, 0.0]
        b = [10.0, 10.0]
        k = rbf_kernel(a, b, length_scale=1.0)
        self.assertLess(k, 0.01)

    def test_gp_predict_empty_training(self):
        mu, sigma2 = gp_predict([], [], [0.5])
        self.assertAlmostEqual(mu, 0.0)
        self.assertGreater(sigma2, 0)

    def test_expected_improvement_positive(self):
        ei = expected_improvement(mu=1.0, sigma2=0.5, best_y=0.5)
        self.assertGreater(ei, 0)

    def test_expected_improvement_zero_sigma(self):
        ei = expected_improvement(mu=1.0, sigma2=0.0, best_y=0.5)
        self.assertAlmostEqual(ei, 0.0)

    def test_optimiser_finds_maximum(self):
        """Test that BO can find the maximum of a simple 1D function."""
        # Objective: -(x - 0.7)^2 + 1.0, maximum at x=0.7
        def objective(params):
            x = params[0]
            return -(x - 0.7) ** 2 + 1.0

        opt = BayesianOptimiser(
            param_bounds=[(0.0, 1.0)],
            objective_fn=objective,
            n_initial=5,
            n_iterations=15,
            seed=42,
        )
        best_params, best_val = opt.optimise()

        # Should be close to 0.7
        self.assertAlmostEqual(best_params[0], 0.7, places=1)
        self.assertGreater(best_val, 0.9)

    def test_optimiser_invalid_bounds_raises(self):
        with self.assertRaises(ValueError):
            BayesianOptimiser(
                param_bounds=[(1.0, 0.0)],  # invalid: lo >= hi
                objective_fn=lambda x: 0,
            )

    def test_optimiser_empty_bounds_raises(self):
        with self.assertRaises(ValueError):
            BayesianOptimiser(
                param_bounds=[],
                objective_fn=lambda x: 0,
            )


class TestFullAlignmentReport(unittest.TestCase):
    def test_report_keys(self):
        mems = [Memory(id="a", text="python flask web", embedding=[1, 0, 0])]
        report = full_alignment_report(
            response_text="python flask web backend",
            memories=mems,
            user_preference=0.5,
        )
        expected_keys = {
            "user_preference", "token_dependence",
            "embedding_dependence", "combined_dependence", "alignment_error",
        }
        self.assertEqual(set(report.keys()), expected_keys)

    def test_report_values_in_range(self):
        mems = [Memory(id="a", text="python flask web", embedding=[1, 0, 0])]
        report = full_alignment_report(
            response_text="python flask",
            memories=mems,
            user_preference=0.8,
        )
        for key, val in report.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)


class TestFullInformationLoss(unittest.TestCase):
    def test_no_loss_when_all_retained(self):
        mems = [
            Memory(id="a", text="python flask", embedding=[1, 0]),
            Memory(id="b", text="java spring", embedding=[0, 1]),
        ]
        report = full_information_loss_report(mems, mems)
        self.assertAlmostEqual(report["memory_reduction"], 0.0)
        self.assertAlmostEqual(report["token_reduction"], 0.0)

    def test_total_loss_when_none_retained(self):
        mems = [Memory(id="a", text="python flask", embedding=[1, 0])]
        report = full_information_loss_report(mems, [])
        self.assertAlmostEqual(report["memory_reduction"], 1.0)
        self.assertAlmostEqual(report["token_reduction"], 1.0)


if __name__ == "__main__":
    unittest.main()
