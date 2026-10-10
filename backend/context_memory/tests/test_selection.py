"""
Tests for MMR, lexicographic ranking, and knapsack selection.
"""

import unittest
import math
from context_memory.memory_store import Memory
from context_memory.retrieval import DenseRetriever, cosine_similarity
from context_memory.mmr import mmr_select
from context_memory.ranking import lexicographic_rank
from context_memory.knapsack import knapsack_select, greedy_knapsack


def _make_indexed_memories():
    """Create a small set of memories with embeddings for testing."""
    memories = [
        Memory(id="a", text="python flask web backend api development", importance=0.9),
        Memory(id="b", text="python flask backend web application server", importance=0.7),
        Memory(id="c", text="java enterprise application server deployment", importance=0.6),
        Memory(id="d", text="database postgresql schema design tables", importance=0.8),
        Memory(id="e", text="react frontend javascript user interface", importance=0.5),
    ]
    retriever = DenseRetriever()
    retriever.index(memories)
    query_vec = retriever.vectoriser.transform("python flask web backend")
    return memories, query_vec


class TestMMR(unittest.TestCase):
    def setUp(self):
        self.memories, self.query_vec = _make_indexed_memories()

    def test_mmr_selects_correct_count(self):
        selected = mmr_select(self.query_vec, self.memories, top_k=3, lambda_param=0.5)
        self.assertEqual(len(selected), 3)

    def test_mmr_penalises_redundancy(self):
        # With lambda=0.5, the two near-duplicate python/flask memories
        # should not both be selected in top 2
        selected = mmr_select(self.query_vec, self.memories, top_k=3, lambda_param=0.3)
        ids = [m.id for m in selected]
        # At least one non-python memory should appear in top 3 due to diversity
        non_python = [mid for mid in ids if mid not in ("a", "b")]
        self.assertTrue(len(non_python) >= 1,
                        f"Expected diversity but got {ids}")

    def test_high_lambda_favours_relevance(self):
        # With lambda=1.0, pure relevance, both python memories should be top
        selected = mmr_select(self.query_vec, self.memories, top_k=2, lambda_param=1.0)
        ids = {m.id for m in selected}
        self.assertTrue(ids.issubset({"a", "b"}),
                        f"Expected most relevant but got {ids}")

    def test_mmr_no_embeddings(self):
        bare = [Memory(id="x", text="test")]
        selected = mmr_select(self.query_vec, bare, top_k=1)
        self.assertEqual(len(selected), 0)

    def test_invalid_lambda_raises(self):
        with self.assertRaises(ValueError):
            mmr_select(self.query_vec, self.memories, lambda_param=1.5)

    def test_invalid_top_k_raises(self):
        with self.assertRaises(ValueError):
            mmr_select(self.query_vec, self.memories, top_k=0)


class TestLexicographicRanking(unittest.TestCase):
    def setUp(self):
        self.memories, self.query_vec = _make_indexed_memories()

    def test_ranking_returns_all(self):
        ranked = lexicographic_rank(self.memories, query_embedding=self.query_vec)
        self.assertEqual(len(ranked), len(self.memories))

    def test_most_relevant_first(self):
        ranked = lexicographic_rank(self.memories, query_embedding=self.query_vec)
        # First should be most relevant to "python flask web backend"
        top_id = ranked[0][0].id
        self.assertIn(top_id, ("a", "b"))

    def test_scores_contain_all_criteria(self):
        ranked = lexicographic_rank(self.memories, query_embedding=self.query_vec)
        _, scores = ranked[0]
        names = [name for name, _ in scores]
        self.assertIn("relevance", names)
        self.assertIn("importance", names)
        self.assertIn("recency", names)
        self.assertIn("access_frequency", names)


class TestKnapsack(unittest.TestCase):
    def setUp(self):
        self.memories = [
            Memory(id="small_high", text="short text", importance=0.9),
            Memory(id="medium_med", text="this is a medium length text example", importance=0.5),
            Memory(id="large_low", text="this is a much longer text that contains many words "
                   "and takes up significant token budget space in the context", importance=0.3),
        ]

    def test_budget_never_exceeded(self):
        for budget in [5, 10, 20, 50, 100]:
            selected, total_tokens, _ = knapsack_select(
                self.memories, token_budget=budget,
            )
            self.assertLessEqual(total_tokens, budget,
                                 f"Budget {budget} exceeded: {total_tokens}")

    def test_higher_importance_preferred(self):
        selected, _, _ = knapsack_select(self.memories, token_budget=10)
        if selected:
            ids = [m.id for m in selected]
            # small_high has highest importance and fits easily
            self.assertIn("small_high", ids)

    def test_empty_candidates(self):
        selected, tokens, value = knapsack_select([], token_budget=100)
        self.assertEqual(selected, [])
        self.assertEqual(tokens, 0)
        self.assertEqual(value, 0.0)

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            knapsack_select(self.memories, token_budget=0)

    def test_custom_value_fn(self):
        selected, _, total_value = knapsack_select(
            self.memories, token_budget=50,
            value_fn=lambda m: m.access_count,
        )
        # All access counts are 0 by default
        self.assertEqual(total_value, 0.0)

    def test_greedy_respects_budget(self):
        for budget in [5, 10, 20]:
            selected, total_tokens, _ = greedy_knapsack(
                self.memories, token_budget=budget,
            )
            self.assertLessEqual(total_tokens, budget)

    def test_dp_vs_greedy_both_respect_budget(self):
        budget = 15
        dp_sel, dp_tok, dp_val = knapsack_select(self.memories, budget)
        gr_sel, gr_tok, gr_val = greedy_knapsack(self.memories, budget)
        self.assertLessEqual(dp_tok, budget)
        self.assertLessEqual(gr_tok, budget)
        # DP should be >= greedy (optimal)
        self.assertGreaterEqual(dp_val, gr_val - 1e-9)


if __name__ == "__main__":
    unittest.main()
