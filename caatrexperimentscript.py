"""
OT-TRPO (Optimal Transport TRPO) implementation with dual formulation
for the differential game. This script is configured to run only the
'caatr' adaptive trust region method and export all resulting data
to text files for analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import copy
import os


class DifferentialGameEnv:
    def __init__(self):
        self.n_agents = 2

    def reward(self, a1, a2):
        """Compute reward for joint actions."""
        # Global optimum at (5,5)
        global_term = np.exp(-0.5 * ((a1 - 5)**2 / 1.0 + (a2 - 5)**2 / 9.0))
        global_coef = 10.0 / (2 * np.pi * np.sqrt(9.0))

        # Strengthened local optimum at (1,1)
        local_term = np.exp(-0.5 * ((a1 - 1)**2 / 1.0 + (a2 - 1)**2 / 1.0))
        local_coef = 6.5 / (2 * np.pi) # Increased from 5.3

        return global_coef * global_term + local_coef * local_term

class GaussianPolicy:
    """Simple Gaussian policy with mean and std parameters."""
    def __init__(self, mean=1.0, std=1.15):
        self.mean = mean
        self.std = std
        self.mean_history = [mean]
        self.std_history = [std]

    def sample(self, n_samples=1):
        """Sample actions from the policy."""
        samples = np.random.normal(self.mean, self.std, n_samples)
        return np.clip(samples, 0, 7)

    def wasserstein_distance(self, other_policy):
        """1-Wasserstein distance for 1D Gaussians."""
        return np.abs(self.mean - other_policy.mean) + np.abs(self.std - other_policy.std)

    def update(self, new_mean, new_std):
        """Update policy parameters and record history."""
        self.mean = new_mean
        self.std = new_std
        self.mean_history.append(new_mean)
        self.std_history.append(new_std)


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
    OT-TRPO with a selectable adaptive trust region method.
    """
    def __init__(self, env, epsilon=0.1, batch_size=30, critic_lr=0.2,
                 n_iterations=4000, initial_mean=1.5, initial_std=0.5,
                 transport_cost_type='l2', adaptive_method='fixed', caatr_C=0.02):
        self.env = env
        self.n_agents = env.n_agents
        self.epsilon_total = epsilon
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.transport_cost_type = transport_cost_type
        self.adaptive_method = adaptive_method
        self.caatr_C = caatr_C  # Hyperparameter for CAATR method

        self.policies = [GaussianPolicy(mean=initial_mean, std=initial_std) for _ in range(self.n_agents)]
        self.critics = [SimpleCritic(lr=critic_lr) for _ in range(self.n_agents)]

        # --- History Tracking ---
        self.reward_history = []
        self.trajectory = {'a1': [initial_mean], 'a2': [initial_mean]}
        self.lambda_history = [[] for _ in range(self.n_agents)]
        self.wasserstein_history = [[0.0] for _ in range(self.n_agents)]
        self.epsilon_history = [[epsilon] for _ in range(self.n_agents)]

    def _get_adaptive_epsilons(self, batch):
        """
        Calculates the trust region radius for each agent based on the CAATR method.
        """
        if self.adaptive_method != 'caatr':
            return [self.epsilon_total] * self.n_agents

        # --- Coordination-Aware Adaptive Trust Region (CAATR) ---
        if len(self.wasserstein_history[0]) < 2: # Need history
            return [self.epsilon_total] * self.n_agents

        epsilons = []
        for i in range(self.n_agents):
            teammate_drift = sum([self.wasserstein_history[j][-1] for j in range(self.n_agents) if i != j])
            eps = self.caatr_C / (teammate_drift + 1e-8)
            epsilons.append(eps)
        return epsilons

    def transport_cost(self, a1, a2):
        return (a1 - a2)**2 if self.transport_cost_type == 'l2' else np.abs(a1 - a2)

    def collect_batch(self):
        batch = {'actions': [[] for _ in range(self.n_agents)], 'rewards': []}
        for _ in range(self.batch_size):
            actions = [p.sample() for p in self.policies]
            reward = self.env.reward(*actions)
            for i in range(self.n_agents):
                batch['actions'][i].append(actions[i])
            batch['rewards'].append(reward)
        return batch

    def compute_advantages(self, batch, agent_id):
        rewards = np.array(batch['rewards'])
        baseline = self.critics[agent_id].get_baseline()
        advantages = rewards - baseline
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    def solve_dual_problem(self, agent_id, batch, old_policy, epsilon_i):
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
                    return -(advantage_function(a_prime) - lambda_val * self.transport_cost(a_old, a_prime))
                res = minimize_scalar(inner_obj, bounds=(0, 7), method='bounded')
                inner_values.append(-res.fun)
            return lambda_val * epsilon_i + np.mean(inner_values)

        res = minimize_scalar(dual_objective, bounds=(0, 20.0), method='bounded')
        optimal_lambda = res.x

        action_grid = np.linspace(0, 7, 50)
        policy_weights = np.zeros_like(action_grid)
        for i, a_prime in enumerate(action_grid):
            values = [advantage_function(a_prime) - optimal_lambda * self.transport_cost(a_old, a_prime) for a_old in actions]
            policy_weights[i] = np.mean(values)

        policy_weights = np.exp(policy_weights / (old_policy.std**2 + 1e-8))
        policy_weights /= np.sum(policy_weights)

        new_mean = np.sum(action_grid * policy_weights)
        new_var = np.sum(((action_grid - new_mean)**2) * policy_weights)
        new_std = np.sqrt(new_var) if new_var > 0.01 else old_policy.std
        return np.clip(new_mean, 0, 7), np.clip(new_std, 0.1, 3.0), optimal_lambda

    def update_agent_dual(self, agent_id, batch, epsilon_i):
        old_policy = copy.deepcopy(self.policies[agent_id])
        try:
            new_mean, new_std, opt_lambda = self.solve_dual_problem(agent_id, batch, old_policy, epsilon_i)

            temp_policy = GaussianPolicy(new_mean, new_std)
            w_dist = old_policy.wasserstein_distance(temp_policy)

            if w_dist > epsilon_i:
                alpha = epsilon_i / (w_dist + 1e-8)
                new_mean = old_policy.mean + alpha * (new_mean - old_policy.mean)
                new_std = old_policy.std + alpha * (new_std - old_policy.std)

            self.policies[agent_id].update(new_mean, new_std)
            final_w_dist = old_policy.wasserstein_distance(self.policies[agent_id])

            self.lambda_history[agent_id].append(opt_lambda)
            self.wasserstein_history[agent_id].append(final_w_dist)
            self.epsilon_history[agent_id].append(epsilon_i)
        except Exception:
            self.policies[agent_id].update(old_policy.mean, old_policy.std)
            self.lambda_history[agent_id].append(0.0)
            self.wasserstein_history[agent_id].append(0.0)
            self.epsilon_history[agent_id].append(epsilon_i)

    def train(self):
        print(f"Training OT-TRPO with adaptive method: '{self.adaptive_method}'")
        for iteration in range(self.n_iterations):
            batch = self.collect_batch()
            for critic in self.critics: critic.update(batch['rewards'])

            adaptive_epsilons = self._get_adaptive_epsilons(batch)

            for agent_id in range(self.n_agents):
                self.update_agent_dual(agent_id, batch, adaptive_epsilons[agent_id])

            avg_reward = np.mean(batch['rewards'])
            self.reward_history.append(avg_reward)
            self.trajectory['a1'].append(self.policies[0].mean)
            self.trajectory['a2'].append(self.policies[1].mean)

            if iteration > 0 and iteration % 200 == 0:
                 print(f"Iter {iteration:4d}: Avg Reward={avg_reward:.3f}, "
                      f"Actions=({self.policies[0].mean:.2f}, {self.policies[1].mean:.2f}), "
                      f"Epsilons=({adaptive_epsilons[0]:.3f}, {adaptive_epsilons[1]:.3f})")

    # MODIFIED: New function to save data to text files
    def save_data_to_files(self, base_filename="caatr_results"):
        """Saves all historical data to separate .txt files."""
        print(f"\nSaving numerical data to files with base name: {base_filename}")

        # 1. Trajectory Data
        with open(f"{base_filename}_trajectory.txt", "w") as f:
            f.write("iteration,agent1_action_mean,agent2_action_mean\n")
            for i in range(len(self.trajectory['a1'])):
                f.write(f"{i},{self.trajectory['a1'][i]},{self.trajectory['a2'][i]}\n")

        # 2. Reward History
        with open(f"{base_filename}_rewards.txt", "w") as f:
            f.write("iteration,average_reward\n")
            for i, reward in enumerate(self.reward_history):
                f.write(f"{i},{reward}\n")

        # 3. Wasserstein Distance History
        with open(f"{base_filename}_w_distance.txt", "w") as f:
            f.write("iteration,agent1_w_dist,agent2_w_dist\n")
            for i in range(len(self.wasserstein_history[0])):
                f.write(f"{i},{self.wasserstein_history[0][i]},{self.wasserstein_history[1][i]}\n")

        # 4. Adaptive Epsilon History
        with open(f"{base_filename}_epsilons.txt", "w") as f:
            f.write("iteration,agent1_epsilon,agent2_epsilon\n")
            for i in range(len(self.epsilon_history[0])):
                f.write(f"{i},{self.epsilon_history[0][i]},{self.epsilon_history[1][i]}\n")

        # 5. Policy Standard Deviation History
        with open(f"{base_filename}_stds.txt", "w") as f:
            f.write("iteration,agent1_std,agent2_std\n")
            for i in range(len(self.policies[0].std_history)):
                f.write(f"{i},{self.policies[0].std_history[i]},{self.policies[1].std_history[i]}\n")

        print("Data saving complete.")


    def plot_results(self):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        a1_range, a2_range = np.linspace(0, 7, 100), np.linspace(0, 7, 100)
        A1, A2 = np.meshgrid(a1_range, a2_range)
        R = self.env.reward(A1, A2)

        # Plot 1: Trajectory
        ax = axes[0, 0]
        im = ax.contourf(A1, A2, R, levels=50, cmap='viridis')
        ax.plot(self.trajectory['a1'], self.trajectory['a2'], 'r-', lw=2)
        ax.scatter(self.trajectory['a1'][0], self.trajectory['a2'][0], c='lime', s=150, zorder=5, marker='o', edgecolors='k', label='Start')
        ax.scatter(self.trajectory['a1'][-1], self.trajectory['a2'][-1], c='red', s=150, zorder=5, marker='*', edgecolors='w', label='End')
        ax.set_title('Policy Trajectory'); ax.set_xlabel('Agent 1 Action'); ax.set_ylabel('Agent 2 Action'); ax.legend()
        fig.colorbar(im, ax=ax)

        # Plot 2: Learning Curve
        ax = axes[0, 1]
        ax.plot(self.reward_history); ax.set_title('Learning Curve'); ax.set_xlabel('Iteration'); ax.set_ylabel('Average Reward')
        ax.grid(True, alpha=0.3)

        # Plot 3: Agent Actions
        ax = axes[0, 2]
        ax.plot(self.trajectory['a1'], label='Agent 1'); ax.plot(self.trajectory['a2'], label='Agent 2')
        ax.set_title('Agent Actions Over Time'); ax.set_xlabel('Iteration'); ax.set_ylabel('Action Value'); ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Wasserstein Distance
        ax = axes[1, 0]
        ax.plot(self.wasserstein_history[0], label='Agent 1 W-dist'); ax.plot(self.wasserstein_history[1], label='Agent 2 W-dist')
        ax.set_title('Policy Update Distance'); ax.set_xlabel('Iteration'); ax.set_ylabel('Wasserstein Distance'); ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Adaptive Epsilon History
        ax = axes[1, 1]
        ax.plot(self.epsilon_history[0], label='Agent 1 ε'); ax.plot(self.epsilon_history[1], label='Agent 2 ε')
        ax.set_title('Trust Region Radius Over Time'); ax.set_xlabel('Iteration'); ax.set_ylabel('Adaptive Epsilon'); ax.legend()
        ax.set_yscale('log'); ax.grid(True, which="both", ls="-", alpha=0.3)

        # Plot 6: Policy Standard Deviations
        ax = axes[1, 2]
        ax.plot([h for h in self.policies[0].std_history], label='Agent 1 std'); ax.plot([h for h in self.policies[1].std_history], label='Agent 2 std')
        ax.set_title('Policy Standard Deviations'); ax.set_xlabel('Iteration'); ax.set_ylabel('Policy Std'); ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'OT-TRPO Results - Method: {self.adaptive_method.upper()}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'ottrpo_results_{self.adaptive_method}.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    env = DifferentialGameEnv()

    # MODIFIED: Define a single configuration for the CAATR experiment
    config = {'name': 'CAATR', 'method': 'caatr', 'epsilon': 0.1, 'C': 0.02}

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: {config['name']} (method: {config['method']})")
    print(f"{'='*60}")

    ottrpo = OTTRPO(
        env,
        epsilon=config['epsilon'],
        batch_size=30,
        critic_lr=0.2,
        n_iterations=3000,
        initial_mean=1.5,
        initial_std=0.5,
        adaptive_method=config['method'],
        caatr_C=config.get('C', 0.02)
    )

    ottrpo.train()
    ottrpo.plot_results()

    # MODIFIED: Call the new function to save the data
    ottrpo.save_data_to_files(base_filename=f"ottrpo_results_{config['method']}")
