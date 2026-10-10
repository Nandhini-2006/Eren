"""
Tests for MinHash, LSH, and delta encoding.
"""

import unittest
from context_memory.minhash import MinHash, find_near_duplicates
from context_memory.lsh import LSHIndex
from context_memory.delta_encoding import (
    compute_delta, apply_delta, DeltaHistory, compression_ratio, delta_size,
)


class TestMinHash(unittest.TestCase):
    def test_identical_texts_high_similarity(self):
        mh = MinHash(num_hashes=128, shingle_size=2)
        text = "the quick brown fox jumps over the lazy dog"
        sim = mh.similarity(text, text)
        self.assertAlmostEqual(sim, 1.0)

    def test_completely_different_texts_low_similarity(self):
        mh = MinHash(num_hashes=256, shingle_size=2)
        a = "quantum physics electron proton neutron energy"
        b = "cooking recipe pasta tomato garlic olive oil"
        sim = mh.similarity(a, b)
        self.assertLess(sim, 0.3)

    def test_similar_texts_moderate_similarity(self):
        mh = MinHash(num_hashes=256, shingle_size=2)
        a = "python flask web backend api development server"
        b = "python flask web backend application development framework"
        sim = mh.similarity(a, b)
        self.assertGreater(sim, 0.2)

    def test_signature_length(self):
        mh = MinHash(num_hashes=64)
        sig = mh.signature("hello world test")
        self.assertEqual(len(sig), 64)

    def test_shingles(self):
        mh = MinHash(shingle_size=2)
        shingles = mh.shingles("the quick brown fox")
        # "the" is a stop word, so tokens are [quick, brown, fox]
        # 2-shingles: "quick brown", "brown fox"
        self.assertIn("quick brown", shingles)
        self.assertIn("brown fox", shingles)

    def test_invalid_num_hashes_raises(self):
        with self.assertRaises(ValueError):
            MinHash(num_hashes=0)

    def test_find_near_duplicates(self):
        texts = [
            "python flask web backend api development",
            "python flask web backend application framework",
            "quantum physics electron proton energy",
        ]
        pairs = find_near_duplicates(texts, threshold=0.15, num_hashes=256)
        # First two should be similar
        found_01 = any(i == 0 and j == 1 for i, j, _ in pairs)
        self.assertTrue(found_01, f"Expected (0,1) pair, got {pairs}")


class TestLSH(unittest.TestCase):
    def setUp(self):
        self.lsh = LSHIndex(num_hashes=128, bands=16, shingle_size=2)

    def test_insert_and_size(self):
        self.lsh.insert("a", "python flask web")
        self.lsh.insert("b", "java spring boot")
        self.assertEqual(self.lsh.size, 2)

    def test_similar_items_become_candidates(self):
        # Use more bands (lower threshold) to detect similarity
        lsh = LSHIndex(num_hashes=128, bands=64, shingle_size=2)
        lsh.insert("a", "python flask web backend api development server routes endpoints testing deployment")
        lsh.insert("b", "python flask web backend api application development server routes endpoints deployment")
        lsh.insert("c", "quantum physics electron proton neutron energy levels photon wavelength frequency spectrum")
        candidates = lsh.query_candidates("a")
        # a and b share most shingles so should be candidates
        self.assertIn("b", candidates)

    def test_query_text(self):
        lsh = LSHIndex(num_hashes=128, bands=64, shingle_size=2)
        lsh.insert("x", "python flask web backend api development server routes endpoints testing")
        candidates = lsh.query_text("python flask web backend api application server routes endpoints testing")
        # Should find "x" as a candidate
        self.assertIn("x", candidates)


    def test_find_all_candidate_pairs(self):
        self.lsh.insert("a", "hello world test example foo bar")
        self.lsh.insert("b", "hello world test example baz qux")
        pairs = self.lsh.find_all_candidate_pairs()
        # Pairs is a list of (id, id) tuples
        self.assertIsInstance(pairs, list)

    def test_candidate_pair_threshold(self):
        threshold = self.lsh.candidate_pair_threshold()
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)

    def test_nonexistent_item_raises(self):
        with self.assertRaises(KeyError):
            self.lsh.query_candidates("nonexistent")

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            LSHIndex(num_hashes=10, bands=3)  # 10 not divisible by 3


class TestDeltaEncoding(unittest.TestCase):
    def test_identical_texts(self):
        text = "line one\nline two\nline three\n"
        delta = compute_delta(text, text)
        self.assertEqual(apply_delta(text, delta), text)

    def test_simple_update(self):
        old = "line one\nline two\nline three\n"
        new = "line one\nline TWO\nline three\n"
        delta = compute_delta(old, new)
        reconstructed = apply_delta(old, delta)
        self.assertEqual(reconstructed, new)

    def test_insertion(self):
        old = "line one\nline three\n"
        new = "line one\nline two\nline three\n"
        delta = compute_delta(old, new)
        reconstructed = apply_delta(old, delta)
        self.assertEqual(reconstructed, new)

    def test_deletion(self):
        old = "line one\nline two\nline three\n"
        new = "line one\nline three\n"
        delta = compute_delta(old, new)
        reconstructed = apply_delta(old, delta)
        self.assertEqual(reconstructed, new)

    def test_complete_replacement(self):
        old = "entirely different content\n"
        new = "completely new text here\n"
        delta = compute_delta(old, new)
        reconstructed = apply_delta(old, delta)
        self.assertEqual(reconstructed, new)

    def test_empty_to_content(self):
        old = ""
        new = "new content\n"
        delta = compute_delta(old, new)
        reconstructed = apply_delta(old, delta)
        self.assertEqual(reconstructed, new)

    def test_content_to_empty(self):
        old = "some content\n"
        new = ""
        delta = compute_delta(old, new)
        reconstructed = apply_delta(old, delta)
        self.assertEqual(reconstructed, new)

    def test_delta_history_reconstruction(self):
        history = DeltaHistory("version 0\n")
        history.update("version 1\n")
        history.update("version 2\n")
        history.update("version 3\n")

        self.assertEqual(history.version_count, 4)
        self.assertEqual(history.reconstruct(0), "version 0\n")
        self.assertEqual(history.reconstruct(1), "version 1\n")
        self.assertEqual(history.reconstruct(2), "version 2\n")
        self.assertEqual(history.reconstruct(3), "version 3\n")

    def test_delta_history_incremental(self):
        history = DeltaHistory("line one\nline two\n")
        history.update("line one\nline TWO\n")
        history.update("line one\nline TWO\nline three\n")

        self.assertEqual(history.reconstruct(0), "line one\nline two\n")
        self.assertEqual(history.reconstruct(2), "line one\nline TWO\nline three\n")

    def test_compression_ratio_identical(self):
        text = "same text\n"
        cr = compression_ratio(text, text)
        self.assertLessEqual(cr, 1.0)

    def test_invalid_version_raises(self):
        history = DeltaHistory("initial\n")
        with self.assertRaises(ValueError):
            history.reconstruct(5)

    def test_delta_size_positive(self):
        delta = compute_delta("old\n", "new content here\n")
        self.assertGreater(delta_size(delta), 0)


if __name__ == "__main__":
    unittest.main()
