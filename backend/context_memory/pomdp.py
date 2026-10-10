"""
POMDP (Partially Observable Markov Decision Process) for Memory Management.

Implements a simplified POMDP for choosing memory-management actions
(keep, summarise, archive, delete) under uncertainty about the true
utility of each memory.

Components
----------
- **States**: {high_value, medium_value, low_value}
  The unobservable true utility of a memory.
- **Actions**: {keep, summarise, archive, delete}
  Operations the agent can perform on a memory.
- **Observations**: {frequently_accessed, occasionally_accessed, rarely_accessed}
  Observable signals about memory usage.
- **Transition model** T(s' | s, a): how the true utility changes
  after an action.
- **Observation model** O(o | s', a): probability of observing o
  given the new state and action.
- **Reward model** R(s, a): immediate reward for taking action a
  in state s.
- **Belief**: a probability distribution over states, updated via
  Bayes' rule after each (action, observation) pair.
- **Policy**: maps a belief vector to the best action by computing
  the expected reward under the current belief.

This is a finite-horizon, discrete POMDP solved by belief-space
enumeration (tractable because the state/action spaces are tiny).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import copy


# -------------------------------------------------------------------
# State, Action, Observation enumerations
# -------------------------------------------------------------------

STATES = ["high_value", "medium_value", "low_value"]
ACTIONS = ["keep", "summarise", "archive", "delete"]
OBSERVATIONS = ["frequently_accessed", "occasionally_accessed", "rarely_accessed"]


# -------------------------------------------------------------------
# Transition model  T[action][state][next_state]
# -------------------------------------------------------------------

TRANSITIONS: Dict[str, Dict[str, Dict[str, float]]] = {
    "keep": {
        "high_value":   {"high_value": 0.8, "medium_value": 0.15, "low_value": 0.05},
        "medium_value": {"high_value": 0.1, "medium_value": 0.7, "low_value": 0.2},
        "low_value":    {"high_value": 0.05, "medium_value": 0.15, "low_value": 0.8},
    },
    "summarise": {
        "high_value":   {"high_value": 0.5, "medium_value": 0.4, "low_value": 0.1},
        "medium_value": {"high_value": 0.05, "medium_value": 0.6, "low_value": 0.35},
        "low_value":    {"high_value": 0.0, "medium_value": 0.2, "low_value": 0.8},
    },
    "archive": {
        "high_value":   {"high_value": 0.3, "medium_value": 0.4, "low_value": 0.3},
        "medium_value": {"high_value": 0.05, "medium_value": 0.3, "low_value": 0.65},
        "low_value":    {"high_value": 0.0, "medium_value": 0.1, "low_value": 0.9},
    },
    "delete": {
        "high_value":   {"high_value": 0.0, "medium_value": 0.0, "low_value": 1.0},
        "medium_value": {"high_value": 0.0, "medium_value": 0.0, "low_value": 1.0},
        "low_value":    {"high_value": 0.0, "medium_value": 0.0, "low_value": 1.0},
    },
}


# -------------------------------------------------------------------
# Observation model  O[action][next_state][observation]
# -------------------------------------------------------------------

OBSERVATIONS_MODEL: Dict[str, Dict[str, Dict[str, float]]] = {
    "keep": {
        "high_value":   {"frequently_accessed": 0.7, "occasionally_accessed": 0.2, "rarely_accessed": 0.1},
        "medium_value": {"frequently_accessed": 0.2, "occasionally_accessed": 0.5, "rarely_accessed": 0.3},
        "low_value":    {"frequently_accessed": 0.05, "occasionally_accessed": 0.25, "rarely_accessed": 0.7},
    },
    "summarise": {
        "high_value":   {"frequently_accessed": 0.5, "occasionally_accessed": 0.35, "rarely_accessed": 0.15},
        "medium_value": {"frequently_accessed": 0.15, "occasionally_accessed": 0.5, "rarely_accessed": 0.35},
        "low_value":    {"frequently_accessed": 0.05, "occasionally_accessed": 0.2, "rarely_accessed": 0.75},
    },
    "archive": {
        "high_value":   {"frequently_accessed": 0.3, "occasionally_accessed": 0.4, "rarely_accessed": 0.3},
        "medium_value": {"frequently_accessed": 0.1, "occasionally_accessed": 0.3, "rarely_accessed": 0.6},
        "low_value":    {"frequently_accessed": 0.02, "occasionally_accessed": 0.13, "rarely_accessed": 0.85},
    },
    "delete": {
        "high_value":   {"frequently_accessed": 0.01, "occasionally_accessed": 0.01, "rarely_accessed": 0.98},
        "medium_value": {"frequently_accessed": 0.01, "occasionally_accessed": 0.01, "rarely_accessed": 0.98},
        "low_value":    {"frequently_accessed": 0.01, "occasionally_accessed": 0.01, "rarely_accessed": 0.98},
    },
}


# -------------------------------------------------------------------
# Reward model  R[state][action]
# -------------------------------------------------------------------

REWARDS: Dict[str, Dict[str, float]] = {
    "high_value":   {"keep": 10.0, "summarise": 5.0, "archive": -2.0, "delete": -20.0},
    "medium_value": {"keep": 3.0,  "summarise": 4.0, "archive": 2.0,  "delete": -5.0},
    "low_value":    {"keep": -1.0, "summarise": 0.5, "archive": 3.0,  "delete": 2.0},
}


# -------------------------------------------------------------------
# Belief representation and update
# -------------------------------------------------------------------

class Belief:
    """A probability distribution over POMDP states."""

    def __init__(self, probs: Optional[Dict[str, float]] = None) -> None:
        if probs is None:
            # Uniform prior
            n = len(STATES)
            self.probs = {s: 1.0 / n for s in STATES}
        else:
            self.probs = dict(probs)
        self._normalise()

    def _normalise(self) -> None:
        total = sum(self.probs.values())
        if total <= 0:
            raise ValueError("Belief probabilities sum to zero or negative")
        self.probs = {s: p / total for s, p in self.probs.items()}

    def update(self, action: str, observation: str) -> "Belief":
        """Bayesian belief update given an action and observation.

        b'(s') ∝ O(o | s', a) * Σ_s T(s' | s, a) * b(s)

        Returns a new Belief object (does not modify self).
        """
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        if observation not in OBSERVATIONS:
            raise ValueError(f"Unknown observation: {observation}")

        new_probs: Dict[str, float] = {}
        for s_prime in STATES:
            # Sum over previous states
            transition_sum = sum(
                TRANSITIONS[action][s][s_prime] * self.probs[s]
                for s in STATES
            )
            obs_prob = OBSERVATIONS_MODEL[action][s_prime][observation]
            new_probs[s_prime] = obs_prob * transition_sum

        return Belief(new_probs)

    def expected_reward(self, action: str) -> float:
        """Compute E[R(s, a)] under the current belief."""
        return sum(self.probs[s] * REWARDS[s][action] for s in STATES)

    def most_likely_state(self) -> str:
        """Return the state with the highest belief probability."""
        return max(self.probs, key=lambda s: self.probs[s])

    def entropy(self) -> float:
        """Shannon entropy of the belief distribution (in nats)."""
        import math
        return -sum(
            p * math.log(p) if p > 0 else 0.0
            for p in self.probs.values()
        )

    def copy(self) -> "Belief":
        return Belief(dict(self.probs))

    def __repr__(self) -> str:
        probs_str = ", ".join(f"{s}: {p:.3f}" for s, p in self.probs.items())
        return f"Belief({probs_str})"


# -------------------------------------------------------------------
# Policy
# -------------------------------------------------------------------

def best_action(belief: Belief) -> Tuple[str, float]:
    """Select the action with the highest expected reward.

    Returns (action, expected_reward).
    """
    best_a = ""
    best_er = float("-inf")
    for a in ACTIONS:
        er = belief.expected_reward(a)
        if er > best_er:
            best_er = er
            best_a = a
    return best_a, best_er


def simulate_trajectory(
    initial_belief: Belief,
    observations: List[str],
    action_override: Optional[List[str]] = None,
) -> List[Dict]:
    """Simulate a sequence of belief updates and action selections.

    Parameters
    ----------
    initial_belief : Belief
        Starting belief state.
    observations : list of str
        Observed signals at each time step.
    action_override : list of str, optional
        If provided, forces actions instead of using the policy.

    Returns
    -------
    list of dict
        Each entry contains 'step', 'belief_before', 'action',
        'expected_reward', 'observation', and 'belief_after'.
    """
    trajectory: List[Dict] = []
    belief = initial_belief.copy()

    for t, obs in enumerate(observations):
        if action_override and t < len(action_override):
            action = action_override[t]
        else:
            action, _ = best_action(belief)

        er = belief.expected_reward(action)
        belief_before = belief.copy()
        belief = belief.update(action, obs)

        trajectory.append({
            "step": t,
            "belief_before": dict(belief_before.probs),
            "action": action,
            "expected_reward": er,
            "observation": obs,
            "belief_after": dict(belief.probs),
        })

    return trajectory
