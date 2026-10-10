"""
Tests for retrieval, RRF, and HNSW modules.
"""

import unittest
import math
from context_memory.memory_store import Memory, MemoryStore
from context_memory.retrieval import (
    DenseRetriever, TfidfVectoriser, cosine_similarity, tokenise,
)
from context_memory.hnsw_index import HNSWIndex
from context_memory.rrf import reciprocal_rank_fusion, rrf_from_memory_rankings


class TestTokenise(unittest.TestCase):
    def test_basic_tokenisation(self):
        tokens = tokenise("The quick brown fox jumps")
        self.assertNotIn("the", tokens)  # stop word removed
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)
        self.assertIn("jumps", tokens)

    def test_empty_string(self):
        self.assertEqual(tokenise(""), [])

    def test_punctuation_removal(self):
        tokens = tokenise("hello, world! foo-bar.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)


class TestTfidfVectoriser(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "python is great for data science",
            "java is used for enterprise applications",
            "python and java are programming languages",
        ]
        self.vec = TfidfVectoriser().fit(self.docs)

    def test_fit_builds_vocabulary(self):
        self.assertTrue(len(self.vec.vocab) > 0)
        self.assertEqual(len(self.vec.idf), len(self.vec.vocab))

    def test_transform_returns_correct_dimension(self):
        v = self.vec.transform("python programming")
        self.assertEqual(len(v), len(self.vec.vocab))

    def test_self_similarity_is_one(self):
        v = self.vec.transform(self.docs[0])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_fit_on_empty_raises(self):
        with self.assertRaises(ValueError):
            TfidfVectoriser().fit([])

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TfidfVectoriser().transform("test")


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        self.assertEqual(cosine_similarity(a, b), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        self.memories = [
            Memory(id="a", text="python backend flask web development"),
            Memory(id="b", text="java enterprise application server"),
            Memory(id="c", text="python data science machine learning"),
        ]
        self.retriever = DenseRetriever()
        self.retriever.index(self.memories)

    def test_query_returns_results(self):
        results = self.retriever.query("python flask", top_k=2)
        self.assertEqual(len(results), 2)

    def test_most_relevant_first(self):
        results = self.retriever.query("python flask web", top_k=3)
        # Memory "a" should be most relevant
        self.assertEqual(results[0][0].id, "a")

    def test_scores_are_descending(self):
        results = self.retriever.query("python", top_k=3)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_respected(self):
        results = self.retriever.query("test", top_k=1)
        self.assertLessEqual(len(results), 1)

    def test_invalid_top_k_raises(self):
        with self.assertRaises(ValueError):
            self.retriever.query("test", top_k=0)

    def test_index_empty_raises(self):
        with self.assertRaises(ValueError):
            DenseRetriever().index([])


class TestHNSWIndex(unittest.TestCase):
    def setUp(self):
        self.vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ]
        self.hnsw = HNSWIndex(m=4, ef_construction=20, seed=42)
        for i, v in enumerate(self.vectors):
            self.hnsw.add(v, node_id=i)

    def test_size(self):
        self.assertEqual(self.hnsw.size, 5)

    def test_search_returns_nearest(self):
        results = self.hnsw.search([1.0, 0.0, 0.0], k=1)
        self.assertEqual(results[0][0], 0)  # node_id 0

    def test_search_returns_sorted_by_distance(self):
        results = self.hnsw.search([1.0, 0.0, 0.0], k=3)
        distances = [d for _, d in results]
        self.assertEqual(distances, sorted(distances))

    def test_search_k_respected(self):
        results = self.hnsw.search([0.5, 0.5, 0.0], k=2)
        self.assertLessEqual(len(results), 2)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.hnsw.add([1.0, 2.0])  # wrong dim

    def test_search_empty_index(self):
        empty = HNSWIndex()
        self.assertEqual(empty.search([1.0, 0.0], k=5), [])

    def test_m_too_small_raises(self):
        with self.assertRaises(ValueError):
            HNSWIndex(m=1)


class TestRRF(unittest.TestCase):
    def test_single_ranking(self):
        ranking = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        result = reciprocal_rank_fusion([ranking], k=60)
        self.assertEqual(result[0][0], "a")

    def test_fusion_of_two_rankings(self):
        r1 = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
        r2 = [("b", 0.8), ("c", 0.6), ("a", 0.2)]
        result = reciprocal_rank_fusion([r1, r2], k=60)
        ids = [item_id for item_id, _ in result]
        # All items should be present
        self.assertEqual(set(ids), {"a", "b", "c"})

    def test_item_in_both_rankings_gets_higher_score(self):
        r1 = [("a", 1.0)]
        r2 = [("a", 1.0)]
        result = reciprocal_rank_fusion([r1, r2], k=60)
        self.assertAlmostEqual(result[0][1], 2.0 / 61)

    def test_empty_rankings_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([])

    def test_negative_k_raises(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[("a", 1.0)]], k=-1)


class TestMemoryStore(unittest.TestCase):
    def test_add_and_get(self):
        store = MemoryStore()
        m = Memory(id="test", text="hello world")
        store.add(m)
        self.assertEqual(store.get("test"), m)

    def test_remove(self):
        store = MemoryStore()
        store.add(Memory(id="x", text="data"))
        self.assertTrue(store.remove("x"))
        self.assertFalse(store.remove("x"))

    def test_update_text(self):
        store = MemoryStore()
        store.add(Memory(id="u", text="old text"))
        updated = store.update_text("u", "new text")
        self.assertEqual(updated.text, "new text")
        self.assertEqual(updated.previous_text, "old text")
        self.assertEqual(updated.version, 2)

    def test_update_nonexistent_raises(self):
        store = MemoryStore()
        with self.assertRaises(KeyError):
            store.update_text("missing", "text")

    def test_token_count(self):
        m = Memory(id="t", text="one two three four five")
        self.assertEqual(m.token_count(), math.ceil(5 * 1.3))

    def test_empty_id_raises(self):
        store = MemoryStore()
        with self.assertRaises(ValueError):
            store.add(Memory(id="", text="test"))


if __name__ == "__main__":
    unittest.main()
