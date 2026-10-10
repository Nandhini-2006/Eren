"""
main.py — End-to-end demonstration of the Context Memory Optimization pipeline.

This script exercises all 14 algorithms in sequence on a small set of
example memories:

 1. Loads example memories into a MemoryStore.
 2. Retrieves relevant memories using dense vector retrieval (TF-IDF).
 3. Retrieves approximate nearest neighbours via HNSW.
 4. Combines rankings using Reciprocal Rank Fusion (RRF).
 5. Reduces redundancy using MMR.
 6. Detects near-duplicates with MinHash and LSH.
 7. Ranks candidates using lexicographic multi-criteria ranking.
 8. Selects memories under a token budget using 0/1 knapsack.
 9. Uses the POMDP to choose a memory-management action.
10. Demonstrates delta encoding for memory updates.
11. Runs Bayesian optimisation on a retrieval objective.
12. Calculates confidence-validation results.
13. Computes memory-dependence alignment error.
14. Estimates information loss.

All results are printed as readable intermediate and final outputs.
"""

from __future__ import annotations

import time
import math
from typing import List

from context_memory.memory_store import Memory, MemoryStore
from context_memory.retrieval import DenseRetriever, TfidfVectoriser, cosine_similarity
from context_memory.hnsw_index import HNSWIndex
from context_memory.rrf import reciprocal_rank_fusion, rrf_from_memory_rankings
from context_memory.mmr import mmr_select
from context_memory.ranking import lexicographic_rank, format_ranking
from context_memory.knapsack import knapsack_select
from context_memory.pomdp import Belief, best_action, simulate_trajectory
from context_memory.minhash import MinHash, find_near_duplicates
from context_memory.lsh import LSHIndex
from context_memory.delta_encoding import (
    DeltaHistory, compute_delta, apply_delta, compression_ratio,
)
from context_memory.bayesian_optimization import BayesianOptimiser
from context_memory.confidence_validation import composite_confidence
from context_memory.memory_alignment import full_alignment_report
from context_memory.information_loss import full_information_loss_report
from context_memory.evaluation import (
    Timer, selection_quality_report, redundancy_rate, token_budget_utilisation,
)


# ===================================================================
# Utility
# ===================================================================

def header(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ===================================================================
# Example memories
# ===================================================================

NOW = 1_700_000_000.0  # Fixed timestamp for reproducibility

EXAMPLE_MEMORIES = [
    Memory(
        id="mem_001",
        text="The user prefers Python for backend development and uses Flask for web APIs.",
        importance=0.9,
        created_at=NOW - 86400 * 5,
        last_accessed=NOW - 3600,
        access_count=12,
        tags=["preference", "python", "backend"],
    ),
    Memory(
        id="mem_002",
        text="The project deadline is December 15th. The user wants weekly progress reports.",
        importance=0.95,
        created_at=NOW - 86400 * 10,
        last_accessed=NOW - 7200,
        access_count=8,
        tags=["deadline", "project"],
    ),
    Memory(
        id="mem_003",
        text="The user likes Python programming and prefers Flask framework for building backends.",
        importance=0.6,
        created_at=NOW - 86400 * 3,
        last_accessed=NOW - 86400,
        access_count=3,
        tags=["preference", "python"],
    ),
    Memory(
        id="mem_004",
        text="Database schema uses PostgreSQL with tables for users, orders, and products.",
        importance=0.8,
        created_at=NOW - 86400 * 7,
        last_accessed=NOW - 43200,
        access_count=6,
        tags=["database", "schema"],
    ),
    Memory(
        id="mem_005",
        text="The CI/CD pipeline runs on GitHub Actions with automated testing and deployment.",
        importance=0.7,
        created_at=NOW - 86400 * 14,
        last_accessed=NOW - 86400 * 4,
        access_count=2,
        tags=["devops", "ci-cd"],
    ),
    Memory(
        id="mem_006",
        text="Authentication is handled by JWT tokens with a 24-hour expiry and refresh tokens.",
        importance=0.85,
        created_at=NOW - 86400 * 2,
        last_accessed=NOW - 1800,
        access_count=10,
        tags=["security", "auth"],
    ),
    Memory(
        id="mem_007",
        text="The user mentioned that code reviews should happen before merging pull requests.",
        importance=0.5,
        created_at=NOW - 86400 * 20,
        last_accessed=NOW - 86400 * 10,
        access_count=1,
        tags=["process", "review"],
    ),
    Memory(
        id="mem_008",
        text="Error logging uses structured JSON format and is sent to a centralised logging service.",
        importance=0.65,
        created_at=NOW - 86400 * 6,
        last_accessed=NOW - 86400 * 2,
        access_count=4,
        tags=["logging", "monitoring"],
    ),
    Memory(
        id="mem_009",
        text="The frontend is built with React and communicates with the backend via REST APIs.",
        importance=0.75,
        created_at=NOW - 86400 * 8,
        last_accessed=NOW - 86400 * 1,
        access_count=5,
        tags=["frontend", "react"],
    ),
    Memory(
        id="mem_010",
        text="Performance testing showed the API handles 500 requests per second under load.",
        importance=0.6,
        created_at=NOW - 86400 * 4,
        last_accessed=NOW - 86400 * 3,
        access_count=2,
        tags=["performance", "testing"],
    ),
]


# ===================================================================
# Main demonstration
# ===================================================================

def main() -> None:
    print("Context Memory Optimization — End-to-End Demonstration")
    print(f"Timestamp (fixed): {NOW}")

    # ------------------------------------------------------------------
    # Step 1: Load memories
    # ------------------------------------------------------------------
    header("Step 1: Load Example Memories")
    store = MemoryStore()
    for mem in EXAMPLE_MEMORIES:
        store.add(mem)
    print(f"Loaded {store.size()} memories into MemoryStore.")
    for m in store.all():
        print(f"  [{m.id}] tokens={m.token_count():>3}  importance={m.importance:.2f}  "
              f"accesses={m.access_count}  '{m.text[:60]}...'")

    query = "What programming language and framework does the user prefer for backend?"
    print(f"\nQuery: \"{query}\"")

    # ------------------------------------------------------------------
    # Step 2: Dense vector retrieval
    # ------------------------------------------------------------------
    header("Step 2: Dense Vector Retrieval (TF-IDF)")
    with Timer("dense_retrieval") as t_dense:
        retriever = DenseRetriever()
        retriever.index(store.all())
        dense_results = retriever.query(query, top_k=5)
    print(f"Top-5 by cosine similarity (TF-IDF, lexical):")
    for mem, score in dense_results:
        print(f"  {mem.id}: score={score:.4f}  '{mem.text[:60]}...'")
    print(f"  Time: {t_dense.elapsed:.4f}s")

    # ------------------------------------------------------------------
    # Step 3: HNSW approximate nearest-neighbour search
    # ------------------------------------------------------------------
    header("Step 3: HNSW Index Search")
    with Timer("hnsw") as t_hnsw:
        hnsw = HNSWIndex(m=8, ef_construction=50, seed=42)
        all_mems = store.all()
        for mem in all_mems:
            if mem.embedding is not None:
                hnsw.add(mem.embedding, node_id=int(mem.id.split("_")[1]))
        query_vec = retriever.vectoriser.transform(query)
        hnsw_results = hnsw.search(query_vec, k=5, ef=30)

    id_map = {int(m.id.split("_")[1]): m for m in all_mems}
    hnsw_ranked = []
    print(f"Top-5 by HNSW (Euclidean distance):")
    for node_id, dist in hnsw_results:
        mem = id_map.get(node_id)
        if mem:
            # Convert distance to similarity-like score for RRF
            sim = 1.0 / (1.0 + dist)
            hnsw_ranked.append((mem, sim))
            print(f"  {mem.id}: distance={dist:.4f}  '{mem.text[:60]}...'")
    print(f"  Time: {t_hnsw.elapsed:.4f}s")

    # ------------------------------------------------------------------
    # Step 4: RRF fusion
    # ------------------------------------------------------------------
    header("Step 4: Reciprocal Rank Fusion (RRF)")
    rrf_results = rrf_from_memory_rankings([dense_results, hnsw_ranked], k=60)
    print(f"Fused ranking (dense + HNSW):")
    for mid, score in rrf_results:
        print(f"  {mid}: rrf_score={score:.6f}")

    # Build candidate list from RRF order
    rrf_mem_order = []
    mem_lookup = {m.id: m for m in all_mems}
    for mid, _ in rrf_results:
        if mid in mem_lookup:
            rrf_mem_order.append(mem_lookup[mid])

    # ------------------------------------------------------------------
    # Step 5: MMR diversity selection
    # ------------------------------------------------------------------
    header("Step 5: Maximal Marginal Relevance (MMR)")
    with Timer("mmr") as t_mmr:
        mmr_selected = mmr_select(
            query_embedding=query_vec,
            candidates=rrf_mem_order,
            top_k=5,
            lambda_param=0.6,
        )
    print(f"MMR-selected memories (lambda=0.6):")
    for mem in mmr_selected:
        print(f"  {mem.id}: '{mem.text[:60]}...'")
    print(f"  Time: {t_mmr.elapsed:.4f}s")

    # ------------------------------------------------------------------
    # Step 5b: MinHash near-duplicate detection
    # ------------------------------------------------------------------
    header("Step 5b: MinHash Near-Duplicate Detection")
    texts = [m.text for m in all_mems]
    dups = find_near_duplicates(texts, threshold=0.2, num_hashes=128)
    print(f"Near-duplicate pairs (Jaccard >= 0.2):")
    for i, j, sim in dups:
        print(f"  ({all_mems[i].id}, {all_mems[j].id}): estimated Jaccard={sim:.4f}")
    if not dups:
        print("  None found at threshold 0.2")

    # ------------------------------------------------------------------
    # Step 5c: LSH candidate detection
    # ------------------------------------------------------------------
    header("Step 5c: LSH Candidate Pair Detection")
    lsh = LSHIndex(num_hashes=128, bands=16, shingle_size=2)
    for mem in all_mems:
        lsh.insert(mem.id, mem.text)
    lsh_pairs = lsh.find_all_candidate_pairs()
    print(f"LSH candidate pairs (threshold ~ {lsh.candidate_pair_threshold():.3f}):")
    for a, b in lsh_pairs:
        print(f"  ({a}, {b})")
    if not lsh_pairs:
        print("  None found")

    # ------------------------------------------------------------------
    # Step 6: Lexicographic ranking
    # ------------------------------------------------------------------
    header("Step 6: Lexicographic Multi-Criteria Ranking")
    with Timer("ranking") as t_rank:
        ranked = lexicographic_rank(
            mmr_selected, query_embedding=query_vec, now=NOW,
        )
    print(format_ranking(ranked))
    print(f"  Time: {t_rank.elapsed:.4f}s")

    # ------------------------------------------------------------------
    # Step 7: Knapsack selection under token budget
    # ------------------------------------------------------------------
    header("Step 7: 0/1 Knapsack Selection (Token Budget)")
    token_budget = 50
    print(f"Token budget: {token_budget}")

    def value_fn(m: Memory) -> float:
        """Combined value for knapsack: importance + relevance."""
        if m.embedding is not None:
            return m.importance + cosine_similarity(m.embedding, query_vec)
        return m.importance

    with Timer("knapsack") as t_knap:
        selected, total_tokens, total_value = knapsack_select(
            candidates=[m for m, _ in ranked],
            token_budget=token_budget,
            value_fn=value_fn,
        )
    print(f"Selected {len(selected)} memories ({total_tokens} tokens, value={total_value:.4f}):")
    for mem in selected:
        print(f"  {mem.id}: tokens={mem.token_count()}, importance={mem.importance:.2f}  "
              f"'{mem.text[:50]}...'")
    assert total_tokens <= token_budget, "Token budget violated!"
    print(f"  Budget enforced: {total_tokens} <= {token_budget}")
    print(f"  Time: {t_knap.elapsed:.4f}s")

    # ------------------------------------------------------------------
    # Step 8: POMDP memory-management decision
    # ------------------------------------------------------------------
    header("Step 8: POMDP Memory-Management Decision")
    # Demonstrate with a memory that has low access
    belief = Belief({"high_value": 0.2, "medium_value": 0.5, "low_value": 0.3})
    print(f"Initial belief: {belief}")

    action, er = best_action(belief)
    print(f"Best action: {action} (expected reward: {er:.2f})")

    # Simulate a trajectory
    observations = ["occasionally_accessed", "rarely_accessed", "rarely_accessed"]
    trajectory = simulate_trajectory(belief, observations)
    print(f"\nTrajectory simulation ({len(observations)} steps):")
    for step in trajectory:
        print(f"  Step {step['step']}: action={step['action']}, "
              f"obs={step['observation']}, E[R]={step['expected_reward']:.2f}")
        b_after = step['belief_after']
        print(f"    Belief after: " +
              ", ".join(f"{s}: {p:.3f}" for s, p in b_after.items()))

    # ------------------------------------------------------------------
    # Step 9: Delta encoding
    # ------------------------------------------------------------------
    header("Step 9: Delta Encoding for Memory Updates")
    original_text = (
        "The project deadline is December 15th.\n"
        "The user wants weekly progress reports.\n"
        "Status: in progress.\n"
    )
    updated_text = (
        "The project deadline is December 20th.\n"
        "The user wants weekly progress reports.\n"
        "Status: completed.\n"
        "Final review scheduled for December 22nd.\n"
    )

    history = DeltaHistory(original_text)
    delta = history.update(updated_text)
    print(f"Original text ({len(original_text)} chars):")
    print(f"  {original_text.strip()}")
    print(f"Updated text ({len(updated_text)} chars):")
    print(f"  {updated_text.strip()}")
    print(f"Delta operations: {delta}")
    reconstructed = history.reconstruct(1)
    assert reconstructed == updated_text, "Delta reconstruction failed!"
    print(f"Reconstruction verified: {'PASS' if reconstructed == updated_text else 'FAIL'}")
    cr = compression_ratio(original_text, updated_text)
    print(f"Compression ratio: {cr:.4f} ({'saves space' if cr < 1.0 else 'no saving'})")

    # ------------------------------------------------------------------
    # Step 10: Bayesian optimisation
    # ------------------------------------------------------------------
    header("Step 10: Bayesian Optimisation")
    print("Objective: tune MMR lambda to maximise a retrieval quality metric.")

    def bo_objective(params: List[float]) -> float:
        """Objective function: MMR selection quality as a function of lambda.

        We measure quality as (precision * mean_relevance - redundancy_penalty).
        """
        lam = params[0]
        mmr_sel = mmr_select(
            query_embedding=query_vec,
            candidates=rrf_mem_order,
            top_k=5,
            lambda_param=lam,
        )
        if not mmr_sel:
            return 0.0

        # Mean relevance
        relevances = []
        for m in mmr_sel:
            if m.embedding is not None:
                relevances.append(cosine_similarity(m.embedding, query_vec))
        mean_rel = sum(relevances) / len(relevances) if relevances else 0.0

        # Redundancy penalty
        red = redundancy_rate(mmr_sel, threshold=0.6)

        return mean_rel - 0.5 * red

    optimiser = BayesianOptimiser(
        param_bounds=[(0.1, 0.95)],
        objective_fn=bo_objective,
        n_initial=5,
        n_iterations=15,
        n_candidates=50,
        seed=42,
    )

    with Timer("bayesian_opt") as t_bo:
        best_params, best_val = optimiser.optimise()

    print(f"Best lambda: {best_params[0]:.4f}")
    print(f"Best objective value: {best_val:.4f}")
    print(f"Evaluations: {len(optimiser.history)}")
    print(f"  Time: {t_bo.elapsed:.4f}s")

    # Show a few history points
    subheader("Optimisation history (first 5 and last 5)")
    for entry in optimiser.history[:5]:
        print(f"  lambda={entry['params'][0]:.4f}  value={entry['value']:.4f}")
    if len(optimiser.history) > 10:
        print("  ...")
    for entry in optimiser.history[-5:]:
        print(f"  lambda={entry['params'][0]:.4f}  value={entry['value']:.4f}")

    # ------------------------------------------------------------------
    # Step 11: Confidence validation
    # ------------------------------------------------------------------
    header("Step 11: Confidence-Based Validation")
    conf = composite_confidence(
        query=query,
        query_embedding=query_vec,
        memories=selected,
    )
    print(f"Evidence density:  {conf['evidence_density']:.4f}")
    print(f"Agreement score:   {conf['agreement_score']:.4f}")
    print(f"Coverage score:    {conf['coverage_score']:.4f}")
    print(f"Composite (heuristic): {conf['composite']:.4f}")
    print(f"Interpretation: {conf['interpretation']}")

    # ------------------------------------------------------------------
    # Step 12: Memory-dependence alignment error
    # ------------------------------------------------------------------
    header("Step 12: Memory-Dependence Alignment Error")
    user_preference = 0.7  # User wants high memory dependence
    # Simulate a response that uses memory content
    simulated_response = (
        "The user prefers Python for backend development and typically uses "
        "Flask as their web framework of choice."
    )
    alignment = full_alignment_report(
        response_text=simulated_response,
        memories=selected,
        user_preference=user_preference,
        vectoriser=retriever.vectoriser,
    )
    print(f"User preference:       {alignment['user_preference']:.4f}")
    print(f"Token dependence:      {alignment['token_dependence']:.4f}")
    print(f"Embedding dependence:  {alignment['embedding_dependence']:.4f}")
    print(f"Combined dependence:   {alignment['combined_dependence']:.4f}")
    print(f"Alignment error:       {alignment['alignment_error']:.4f}")

    # ------------------------------------------------------------------
    # Step 13: Information loss estimation
    # ------------------------------------------------------------------
    header("Step 13: Information-Loss Estimation")
    info_loss = full_information_loss_report(
        original_memories=all_mems,
        retained_memories=selected,
        vectoriser=retriever.vectoriser,
    )
    print(f"Token loss:       {info_loss['token_loss']:.4f}")
    print(f"Coverage loss:    {info_loss['coverage_loss']:.4f}")
    print(f"Embedding loss:   {info_loss['embedding_loss']:.4f}")
    print(f"Memory reduction: {info_loss['memory_reduction']:.4f}")
    print(f"Token reduction:  {info_loss['token_reduction']:.4f}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    header("Final Summary")
    relevant_ids = {"mem_001", "mem_003"}  # Ground-truth relevant memories
    quality = selection_quality_report(
        selected=selected,
        all_candidates=all_mems,
        relevant_ids=relevant_ids,
        token_budget=token_budget,
    )
    print(f"Selected memories: {[m.id for m in selected]}")
    print(f"Precision:        {quality['precision']:.4f}")
    print(f"Recall:           {quality['recall']:.4f}")
    print(f"F1 Score:         {quality['f1_score']:.4f}")
    print(f"Redundancy rate:  {quality['redundancy_rate']:.4f}")
    print(f"Token usage:      {quality['total_tokens']}/{quality['budget']} "
          f"({quality['utilisation']:.1%})")
    print(f"Within budget:    {quality['within_budget']}")


if __name__ == "__main__":
    main()
