"""
Tests for POMDP belief updates and policy decisions.
"""

import unittest
import math
from context_memory.pomdp import (
    Belief, best_action, simulate_trajectory,
    STATES, ACTIONS, OBSERVATIONS,
    TRANSITIONS, OBSERVATIONS_MODEL, REWARDS,
)


class TestPOMDPModels(unittest.TestCase):
    """Verify that transition and observation models are valid distributions."""

    def test_transitions_sum_to_one(self):
        for action in ACTIONS:
            for state in STATES:
                total = sum(TRANSITIONS[action][state].values())
                self.assertAlmostEqual(total, 1.0, places=5,
                    msg=f"T({action}, {state}) sums to {total}")

    def test_observations_sum_to_one(self):
        for action in ACTIONS:
            for state in STATES:
                total = sum(OBSERVATIONS_MODEL[action][state].values())
                self.assertAlmostEqual(total, 1.0, places=5,
                    msg=f"O({action}, {state}) sums to {total}")

    def test_all_states_have_rewards(self):
        for state in STATES:
            for action in ACTIONS:
                self.assertIn(action, REWARDS[state])


class TestBelief(unittest.TestCase):
    def test_uniform_prior(self):
        b = Belief()
        for s in STATES:
            self.assertAlmostEqual(b.probs[s], 1.0 / len(STATES))

    def test_custom_prior(self):
        b = Belief({"high_value": 0.8, "medium_value": 0.1, "low_value": 0.1})
        self.assertAlmostEqual(b.probs["high_value"], 0.8)

    def test_normalisation(self):
        b = Belief({"high_value": 2.0, "medium_value": 1.0, "low_value": 1.0})
        total = sum(b.probs.values())
        self.assertAlmostEqual(total, 1.0)

    def test_belief_update_produces_valid_distribution(self):
        b = Belief()
        for action in ACTIONS:
            for obs in OBSERVATIONS:
                b_new = b.update(action, obs)
                total = sum(b_new.probs.values())
                self.assertAlmostEqual(total, 1.0, places=5,
                    msg=f"After ({action}, {obs}): sum={total}")

    def test_belief_update_changes_belief(self):
        b = Belief()
        b_new = b.update("keep", "frequently_accessed")
        # After observing frequent access, high_value should increase
        self.assertGreater(b_new.probs["high_value"], b.probs["high_value"])

    def test_belief_update_with_rare_access(self):
        b = Belief()
        b_new = b.update("keep", "rarely_accessed")
        # After observing rare access, low_value should increase
        self.assertGreater(b_new.probs["low_value"], b.probs["low_value"])

    def test_invalid_action_raises(self):
        b = Belief()
        with self.assertRaises(ValueError):
            b.update("invalid_action", "frequently_accessed")

    def test_invalid_observation_raises(self):
        b = Belief()
        with self.assertRaises(ValueError):
            b.update("keep", "invalid_obs")

    def test_entropy_uniform_is_maximal(self):
        b_uniform = Belief()
        b_peaked = Belief({"high_value": 0.9, "medium_value": 0.05, "low_value": 0.05})
        self.assertGreater(b_uniform.entropy(), b_peaked.entropy())

    def test_most_likely_state(self):
        b = Belief({"high_value": 0.1, "medium_value": 0.2, "low_value": 0.7})
        self.assertEqual(b.most_likely_state(), "low_value")


class TestPolicy(unittest.TestCase):
    def test_best_action_for_high_value_belief(self):
        b = Belief({"high_value": 0.95, "medium_value": 0.04, "low_value": 0.01})
        action, er = best_action(b)
        self.assertEqual(action, "keep")
        self.assertGreater(er, 0)

    def test_best_action_for_low_value_belief(self):
        b = Belief({"high_value": 0.01, "medium_value": 0.09, "low_value": 0.9})
        action, er = best_action(b)
        self.assertIn(action, ["archive", "delete"])

    def test_expected_reward_is_finite(self):
        b = Belief()
        for a in ACTIONS:
            er = b.expected_reward(a)
            self.assertTrue(math.isfinite(er))


class TestTrajectory(unittest.TestCase):
    def test_trajectory_length_matches_observations(self):
        b = Belief()
        obs = ["frequently_accessed", "occasionally_accessed", "rarely_accessed"]
        traj = simulate_trajectory(b, obs)
        self.assertEqual(len(traj), 3)

    def test_trajectory_belief_after_is_valid(self):
        b = Belief()
        obs = ["frequently_accessed", "rarely_accessed"]
        traj = simulate_trajectory(b, obs)
        for step in traj:
            total = sum(step["belief_after"].values())
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_trajectory_with_action_override(self):
        b = Belief()
        obs = ["frequently_accessed"]
        traj = simulate_trajectory(b, obs, action_override=["delete"])
        self.assertEqual(traj[0]["action"], "delete")


if __name__ == "__main__":
    unittest.main()
