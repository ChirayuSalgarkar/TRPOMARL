"""
OT-TRPO (Optimal Transport TRPO) with CAATR for a 7-agent differential game.
This script runs the simulation and calculates the final distance from the global optimum.
"""

import numpy as np
import copy
from scipy.optimize import minimize_scalar

class DifferentialGameEnv7Agent:
    """
    Seven-player differential game adapted from the original paper.
    The reward function is extended to 7 dimensions.
    """

    def __init__(self):
        self.n_agents = 7
        self.true_optimum = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    def reward(self, a1, a2, a3, a4, a5, a6, a7):
        """Compute reward for joint actions of seven agents."""
        # Global optimum at (5,5,5,5,5,5,5)
        # Using different variances for each agent to create asymmetry
        global_term = np.exp(-0.5 * (
            (a1 - 5)**2 / 1.0 +
            (a2 - 5)**2 / 9.0 +
            (a3 - 5)**2 / 1.0 +
            (a4 - 5)**2 / 4.0 +
            (a5 - 5)**2 / 2.0 +
            (a6 - 5)**2 / 3.0 +
            (a7 - 5)**2 / 1.5
        ))
        # Normalize by the product of variances
        variance_product = np.sqrt(1.0 * 9.0 * 1.0 * 4.0 * 2.0 * 3.0 * 1.5)
        global_coef = 10.0 / (((2 * np.pi)**(7/2)) * variance_product)

        # Local optimum at (1,1,1,1,1,1,1)
        local_term = np.exp(-0.5 * (
            (a1 - 1)**2 / 1.0 +
            (a2 - 1)**2 / 1.0 +
            (a3 - 1)**2 / 1.0 +
            (a4 - 1)**2 / 1.0 +
            (a5 - 1)**2 / 1.0 +
            (a6 - 1)**2 / 1.0 +
            (a7 - 1)**2 / 1.0
        ))
        local_coef = 6.5 / ((2 * np.pi)**(7/2))

        # Linear bias term for agent 1 (kept small to maintain multi-modality)
        linear_term = 0.0 * a1

        return global_coef * global_term + local_coef * local_term + linear_term

class GaussianPolicy:
    """Simple Gaussian policy with mean and std parameters."""
    def __init__(self, mean=1.0, std=1.15):
        self.mean = mean
        self.std = std

    def sample(self, n_samples=1):
        """Sample actions from the policy."""
        samples = np.random.normal(self.mean, self.std, n_samples)
        return np.clip(samples, 0, 7)

    def wasserstein_distance(self, other_policy):
        """1-Wasserstein distance for 1D Gaussians."""
        return np.abs(self.mean - other_policy.mean) + np.abs(self.std - other_policy.std)

    def update(self, new_mean, new_std):
        """Update policy parameters."""
        self.mean = new_mean
        self.std = new_std

class SimpleCritic:
    """A simple baseline critic using an exponential moving average of rewards."""
    def __init__(self, lr=0.2):
        self.lr = lr
        self.baseline = 0.0

    def update(self, rewards):
        """Update the baseline."""
        if len(rewards) > 0:
            self.baseline = (1 - self.lr) * self.baseline + self.lr * np.mean(rewards)

    def get_baseline(self):
        return self.baseline

class OTTRPO:
    """
    OT-TRPO with Coordination-Aware Adaptive Trust Region (CAATR) for 7 agents.
    """
    def __init__(self, env, epsilon=0.1, batch_size=30, critic_lr=0.2,
                 n_iterations=4000, initial_mean=1.5, initial_std=0.5,
                 caatr_C=0.02):
        self.env = env
        self.n_agents = env.n_agents
        self.epsilon_total = epsilon
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.caatr_C = caatr_C

        self.policies = [GaussianPolicy(mean=initial_mean, std=initial_std) for _ in range(self.n_agents)]
        self.critics = [SimpleCritic(lr=critic_lr) for _ in range(self.n_agents)]

        # History Tracking
        self.wasserstein_history = [[0.0] for _ in range(self.n_agents)]

    def _get_adaptive_epsilons(self):
        """Calculates trust regions using the CAATR method."""
        if len(self.wasserstein_history[0]) < 2:
            return [self.epsilon_total] * self.n_agents

        epsilons = []
        for i in range(self.n_agents):
            teammate_drift = sum([self.wasserstein_history[j][-1] for j in range(self.n_agents) if i != j])
            eps = self.caatr_C / (teammate_drift + 1e-8)
            epsilons.append(eps)
        return epsilons

    def collect_batch(self):
        """Collects a batch of experience."""
        batch = {'actions': [[] for _ in range(self.n_agents)], 'rewards': []}
        for _ in range(self.batch_size):
            actions = [p.sample() for p in self.policies]
            reward = self.env.reward(*actions)
            for i in range(self.n_agents):
                batch['actions'][i].append(actions[i])
            batch['rewards'].append(reward)
        return batch

    def compute_advantages(self, batch, agent_id):
        """Computes normalized advantages."""
        rewards = np.array(batch['rewards'])
        baseline = self.critics[agent_id].get_baseline()
        advantages = rewards - baseline
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    def solve_dual_problem(self, agent_id, batch, old_policy, epsilon_i):
        """Solves the dual problem to find the optimal policy update."""
        actions = np.array(batch['actions'][agent_id])
        advantages = self.compute_advantages(batch, agent_id)

        def advantage_function(a_prime):
            if len(actions) == 0: return 0.0
            weights = np.exp(-0.5 * ((actions - a_prime) / old_policy.std)**2)
            weights /= (np.sum(weights) + 1e-10)
            return np.sum(weights * advantages)

        def dual_objective(lambda_val):
            if lambda_val < 0: return 1e10
            inner_values = []
            for a_old in actions:
                def inner_obj(a_prime):
                    return -(advantage_function(a_prime) - lambda_val * np.abs(a_old - a_prime))
                res = minimize_scalar(inner_obj, bounds=(0, 7), method='bounded')
                inner_values.append(-res.fun)
            return lambda_val * epsilon_i + np.mean(inner_values)

        res = minimize_scalar(dual_objective, bounds=(0, 20.0), method='bounded')
        optimal_lambda = res.x

        action_grid = np.linspace(0, 7, 50)
        policy_weights = np.zeros_like(action_grid)
        for i, a_prime in enumerate(action_grid):
            values = [advantage_function(a_prime) - optimal_lambda * np.abs(a_old - a_prime) for a_old in actions]
            policy_weights[i] = np.mean(values)

        policy_weights = np.exp(policy_weights / (old_policy.std**2 + 1e-8))
        policy_weights /= np.sum(policy_weights)

        new_mean = np.sum(action_grid * policy_weights)
        new_var = np.sum(((action_grid - new_mean)**2) * policy_weights)
        new_std = np.sqrt(new_var) if new_var > 0.01 else old_policy.std
        return np.clip(new_mean, 0, 7), np.clip(new_std, 0.1, 3.0)

    def update_agent_dual(self, agent_id, batch, epsilon_i):
        """Updates an agent's policy using the dual formulation."""
        old_policy = copy.deepcopy(self.policies[agent_id])
        try:
            new_mean, new_std = self.solve_dual_problem(agent_id, batch, old_policy, epsilon_i)

            temp_policy = GaussianPolicy(new_mean, new_std)
            w_dist = old_policy.wasserstein_distance(temp_policy)

            if w_dist > epsilon_i:
                alpha = epsilon_i / (w_dist + 1e-8)
                new_mean = old_policy.mean + alpha * (new_mean - old_policy.mean)
                new_std = old_policy.std + alpha * (new_std - old_policy.std)

            self.policies[agent_id].update(new_mean, new_std)
            final_w_dist = old_policy.wasserstein_distance(self.policies[agent_id])
            self.wasserstein_history[agent_id].append(final_w_dist)
        except Exception:
            self.policies[agent_id].update(old_policy.mean, old_policy.std)
            self.wasserstein_history[agent_id].append(0.0)

    def train(self):
        """Main training loop."""
        print(f"Training OT-TRPO with CAATR for {self.n_iterations} iterations...")
        print(f"Number of agents: {self.n_agents}")
        for iteration in range(self.n_iterations):
            batch = self.collect_batch()
            for critic in self.critics: critic.update(batch['rewards'])

            adaptive_epsilons = self._get_adaptive_epsilons()

            for agent_id in range(self.n_agents):
                self.update_agent_dual(agent_id, batch, adaptive_epsilons[agent_id])

            if iteration % 100 == 0:
                actions_mean = [p.mean for p in self.policies]
                actions_str = ", ".join([f"{a:.3f}" for a in actions_mean])
                print(f"Iteration {iteration:4d}: Actions = ({actions_str})")

def main():
    """
    Runs the n-agent OT-TRPO simulation and reports the final distance.
    """
    env = DifferentialGameEnv7Agent()

    # Hyperparameters adjusted for 7 agents
    ottrpo = OTTRPO(
        env,
        epsilon=0.1,
        batch_size=40,  # Increased for more agents
        critic_lr=0.2,
        n_iterations=5000,  # More iterations for convergence
        initial_mean=1.5,
        initial_std=0.5,
        caatr_C=0.18  # Adjusted for more agents
    )

    ottrpo.train()

    final_actions = np.array([p.mean for p in ottrpo.policies])
    distance_from_global = np.linalg.norm(final_actions - env.true_optimum)

    print("\n" + "="*60)
    print("OT-TRPO with CAATR Final Results (7 Agents):")
    actions_str = ", ".join([f"{a:.3f}" for a in final_actions])
    optimum_str = ", ".join([f"{o:.1f}" for o in env.true_optimum])
    print(f"  Final Actions: ({actions_str})")
    print(f"  True Optimum:  ({optimum_str})")
    print(f"  Distance from Global Optimum: {distance_from_global:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
