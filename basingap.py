import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import copy
import os


class DifferentialGameEnv:
    def __init__(self, basin_gap_factor=1.0):
        self.n_agents = 2
        self.basin_gap_factor = basin_gap_factor
        self.global_coef_base = 10.0 / (2 * np.pi * np.sqrt(9.0))
        self.local_coef_base = 6.5 / (2 * np.pi)
        self.local_coef = self.local_coef_base * basin_gap_factor
        self.global_coef = self.global_coef_base

    def reward(self, a1, a2):
        global_term = np.exp(-0.5 * ((a1 - 5)**2 / 1.0 + (a2 - 5)**2 / 9.0))
        local_term = np.exp(-0.5 * ((a1 - 1)**2 / 1.0 + (a2 - 1)**2 / 1.0))
        linear_term = 0.1 * a1
        return self.global_coef * global_term + self.local_coef * local_term + linear_term

    def get_reward_at_optima(self):
        local_reward = self.reward(1.0, 1.0)
        global_reward = self.reward(5.0, 5.0)
        return {
            'local': local_reward,
            'global': global_reward,
            'gap': global_reward - local_reward,
            'ratio': global_reward / local_reward if local_reward > 0 else np.inf
        }


class GaussianPolicy:
    def __init__(self, mean=1.0, std=1.15):
        self.mean = mean
        self.std = std
        self.mean_history = [mean]
        self.std_history = [std]

    def sample(self, n_samples=1):
        samples = np.random.normal(self.mean, self.std, n_samples)
        return np.clip(samples, 0, 7)

    def wasserstein_distance(self, other_policy):
        return np.abs(self.mean - other_policy.mean) + np.abs(self.std - other_policy.std)

    def update(self, new_mean, new_std):
        self.mean = new_mean
        self.std = new_std
        self.mean_history.append(new_mean)
        self.std_history.append(new_std)


class SimpleCritic:
    def __init__(self, lr=0.2):
        self.lr = lr
        self.baseline = 0.0

    def update(self, rewards):
        if len(rewards) > 0:
            self.baseline = (1 - self.lr) * self.baseline + self.lr * np.mean(rewards)

    def get_baseline(self):
        return self.baseline


class OTTRPO:
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
        self.caatr_C = caatr_C

        self.policies = [GaussianPolicy(mean=initial_mean, std=initial_std)
                        for _ in range(self.n_agents)]
        self.critics = [SimpleCritic(lr=critic_lr) for _ in range(self.n_agents)]

        self.reward_history = []
        self.trajectory = {'a1': [initial_mean], 'a2': [initial_mean]}
        self.lambda_history = [[] for _ in range(self.n_agents)]
        self.wasserstein_history = [[0.0] for _ in range(self.n_agents)]
        self.epsilon_history = [[epsilon] for _ in range(self.n_agents)]

        self.converged_to_global = False
        self.convergence_iteration = None

    def _get_adaptive_epsilons(self, batch):
        if self.adaptive_method == 'fixed':
            return [self.epsilon_total] * self.n_agents

        advantages = [self.compute_advantages(batch, i) for i in range(self.n_agents)]
        avg_advantages = np.array([np.mean(adv) for adv in advantages])

        if self.adaptive_method == 'greedy':
            last_w_dist = np.array([self.wasserstein_history[i][-1]
                                    for i in range(self.n_agents)])
            scores = np.abs(avg_advantages) / (last_w_dist + 1e-8)
            scores[np.isnan(scores)] = 0
            if np.sum(scores) < 1e-8:
                return [self.epsilon_total / self.n_agents] * self.n_agents
            normalized_scores = scores / np.sum(scores)
            return normalized_scores * self.epsilon_total

        if self.adaptive_method == 'weighted':
            utilities = np.maximum(0, avg_advantages)
            if np.sum(utilities) < 1e-8:
                return [self.epsilon_total / self.n_agents] * self.n_agents

            lambda_val = np.max(utilities) + 1e-6
            for _ in range(10):
                allocations = np.maximum(0, utilities / lambda_val - 1e-4)
                current_total = np.sum(allocations)
                if abs(current_total - self.epsilon_total) < 1e-5:
                    break
                if current_total < 1e-8:
                    lambda_val *= 0.5
                else:
                    lambda_val *= (current_total / self.epsilon_total)

            final_allocations = np.maximum(0, utilities / lambda_val - 1e-4)
            if np.sum(final_allocations) > 1e-8:
                return final_allocations / np.sum(final_allocations) * self.epsilon_total
            return [self.epsilon_total / self.n_agents] * self.n_agents

        if self.adaptive_method == 'caatr':
            if len(self.wasserstein_history[0]) < 2:
                return [self.epsilon_total] * self.n_agents

            epsilons = []
            for i in range(self.n_agents):
                teammate_drift = sum([self.wasserstein_history[j][-1]
                                     for j in range(self.n_agents) if i != j])
                eps = self.caatr_C / (teammate_drift + 1e-8)
                epsilons.append(eps)
            return epsilons

        raise ValueError(f"Unknown adaptive_method: {self.adaptive_method}")

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
            if len(actions) == 0:
                return 0.0
            weights = np.exp(-0.5 * ((actions - a_prime) / old_policy.std)**2)
            weights /= (np.sum(weights) + 1e-10)
            return np.sum(weights * advantages)

        def dual_objective(lambda_val):
            if lambda_val < 0:
                return 1e10
            inner_values = []
            for a_old in actions:
                def inner_obj(a_prime):
                    return -(advantage_function(a_prime) -
                            lambda_val * self.transport_cost(a_old, a_prime))
                res = minimize_scalar(inner_obj, bounds=(0, 7), method='bounded')
                inner_values.append(-res.fun)
            return lambda_val * epsilon_i + np.mean(inner_values)

        res = minimize_scalar(dual_objective, bounds=(0, 20.0), method='bounded')
        optimal_lambda = res.x

        action_grid = np.linspace(0, 7, 50)
        policy_weights = np.zeros_like(action_grid)
        for i, a_prime in enumerate(action_grid):
            values = [advantage_function(a_prime) -
                     optimal_lambda * self.transport_cost(a_old, a_prime)
                     for a_old in actions]
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
            new_mean, new_std, opt_lambda = self.solve_dual_problem(
                agent_id, batch, old_policy, epsilon_i)

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
        except Exception as e:
            self.policies[agent_id].update(old_policy.mean, old_policy.std)
            self.lambda_history[agent_id].append(0.0)
            self.wasserstein_history[agent_id].append(0.0)
            self.epsilon_history[agent_id].append(epsilon_i)

    def check_convergence(self, iteration):
        a1, a2 = self.policies[0].mean, self.policies[1].mean
        distance_to_global = np.sqrt((a1 - 5.0)**2 + (a2 - 5.0)**2)

        if distance_to_global < 0.5 and not self.converged_to_global:
            self.converged_to_global = True
            self.convergence_iteration = iteration
            return True
        return False

    def train(self):
        for iteration in range(self.n_iterations):
            batch = self.collect_batch()
            for critic in self.critics:
                critic.update(batch['rewards'])

            adaptive_epsilons = self._get_adaptive_epsilons(batch)

            for agent_id in range(self.n_agents):
                self.update_agent_dual(agent_id, batch, adaptive_epsilons[agent_id])

            avg_reward = np.mean(batch['rewards'])
            self.reward_history.append(avg_reward)
            self.trajectory['a1'].append(self.policies[0].mean)
            self.trajectory['a2'].append(self.policies[1].mean)

            self.check_convergence(iteration)


def run_basin_gap_experiment():
    basin_factors = [0.5, 1.0, 1.5, 2.0]
    methods = ['fixed', 'greedy', 'weighted', 'caatr']
    results = {}

    for method in methods:
        results[method] = {}

        for k in basin_factors:
            env = DifferentialGameEnv(basin_gap_factor=k)

            if method in ['greedy', 'weighted']:
                epsilon = 0.2
            else:
                epsilon = 0.1

            ottrpo = OTTRPO(
                env,
                epsilon=epsilon,
                batch_size=30,
                critic_lr=0.2,
                n_iterations=3000,
                initial_mean=1.5,
                initial_std=0.5,
                adaptive_method=method,
                caatr_C=0.02 if method == 'caatr' else 0.02
            )

            ottrpo.train()

            results[method][k] = {
                'converged_to_global': ottrpo.converged_to_global,
                'convergence_iter': ottrpo.convergence_iteration,
                'final_actions': (ottrpo.policies[0].mean, ottrpo.policies[1].mean),
                'final_reward': ottrpo.reward_history[-1],
                'trajectory': copy.deepcopy(ottrpo.trajectory),
                'rewards': copy.deepcopy(ottrpo.reward_history)
            }

    create_basin_gap_plots(results, basin_factors, methods)
    return results


def create_basin_gap_plots(results, basin_factors, methods):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]

        for k in basin_factors:
            rewards = results[method][k]['rewards']
            label = f"k={k:.1f}"
            if results[method][k]['converged_to_global']:
                label += " ✓"
            ax.plot(rewards, label=label, alpha=0.7, linewidth=2)

        ax.set_title(f'{method.upper()} Method - Basin Gap Effect')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        env_test = DifferentialGameEnv(basin_gap_factor=1.0)
        global_reward = env_test.reward(5.0, 5.0)
        ax.axhline(y=global_reward, color='g', linestyle='--', alpha=0.5, label='Global')

    plt.suptitle('Basin Gap Experiment: Effect of Local Optimum Strength', fontsize=16)
    plt.tight_layout()

    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(os.path.join('results', 'basin_gap_comparison.png'), dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(len(basin_factors))
    width = 0.2

    for i, method in enumerate(methods):
        success = [1 if results[method][k]['converged_to_global'] else 0
                  for k in basin_factors]
        ax.bar(x + i*width, success, width, label=method.upper())

    ax.set_xlabel('Basin Gap Factor (k)')
    ax.set_ylabel('Converged to Global (1=Yes, 0=No)')
    ax.set_title('Convergence Success by Basin Gap')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{k:.1f}" for k in basin_factors])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    for i, method in enumerate(methods):
        final_rewards = [results[method][k]['final_reward'] for k in basin_factors]
        ax.plot(basin_factors, final_rewards, marker='o', label=method.upper(), linewidth=2)

    ax.set_xlabel('Basin Gap Factor (k)')
    ax.set_ylabel('Final Average Reward')
    ax.set_title('Final Performance by Basin Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'basin_gap_summary.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    results = run_basin_gap_experiment()
