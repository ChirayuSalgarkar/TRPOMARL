"""
W-MATRPO: Unified multi-agent implementation for all agent configurations
Supports 2, 3, 5, 7, and 9 agent setups as specified in the paper
"""

import numpy as np
import copy
from scipy.optimize import minimize_scalar
from typing import List, Tuple, Dict

class DifferentialGameEnv:
    """
    N-agent differential game environment as described in Section 5.1 of the paper.
    """
    
    def __init__(self, n_agents: int = 2):
        self.n_agents = n_agents
        self.global_optimum = np.array([5.0] * n_agents)
        self.local_optimum = np.array([1.0] * n_agents)
        
        # Set variances as per paper (asymmetric for agents 1 and 2)
        self.variances = np.ones(n_agents)
        if n_agents >= 2:
            self.variances[0] = 1.0  # σ₁ = 1
            self.variances[1] = 3.0  # σ₂ = 3 (creates asymmetry)
        if n_agents >= 3:
            self.variances[2] = 1.0  # σ₃ = 1
            
    def reward(self, actions: np.ndarray) -> float:
        """
        Compute reward for joint actions following Equation (30) in the paper.
        """
        # Global optimum term
        global_dist = np.sum((actions - self.global_optimum)**2 / self.variances)
        global_term = np.exp(-0.5 * global_dist)
        global_coef = 10.0 / (((2 * np.pi)**(self.n_agents/2)) * 
                              np.sqrt(np.prod(self.variances)))
        
        # Local optimum term
        local_dist = np.sum((actions - self.local_optimum)**2)
        local_term = np.exp(-0.5 * local_dist)
        local_coef = 6.5 / ((2 * np.pi)**(self.n_agents/2))
        
        # Linear bias term (as per paper)
        linear_term = 0.1 * actions[0]
        
        return global_coef * global_term + local_coef * local_term + linear_term


class WMATRPO:
    """
    W-MATRPO implementation with CAATR mechanism.
    Follows Algorithm 1 and Algorithm 2 from the paper.
    """
    
    def __init__(self, 
                 env: DifferentialGameEnv,
                 n_agents: int,
                 delta: float = 0.1,
                 batch_size: int = 30,
                 n_iterations: int = 4000,
                 caatr_C: float = None,
                 epsilon_base: float = 1e-8,
                 epsilon_max: float = 0.5,
                 initial_mean: float = 1.5,
                 initial_std: float = 0.5,
                 critic_lr: float = 0.2):
        
        self.env = env
        self.n_agents = n_agents
        self.delta = delta
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        
        # Set CAATR constant based on agent count (Table 2)
        if caatr_C is None:
            caatr_C_values = {2: 0.02, 3: 0.02, 5: 0.02, 7: 0.10, 9: 0.15}
            self.caatr_C = caatr_C_values.get(n_agents, 0.02)
        else:
            self.caatr_C = caatr_C
            
        self.epsilon_base = epsilon_base
        self.epsilon_max = epsilon_max
        
        # Initialize policies
        from gaussian_policy import GaussianPolicy
        self.policies = [GaussianPolicy(initial_mean, initial_std) 
                        for _ in range(n_agents)]
        
        # Initialize critics
        from critic import SimpleCritic
        self.critics = [SimpleCritic(critic_lr) for _ in range(n_agents)]
        
        # History tracking
        self.policy_drift_history = np.zeros((n_agents, 0))
        self.delta_history = []
        self.reward_history = []
        self.trajectory_history = []
        
    def compute_caatr_radius(self, agent_id: int, iteration: int) -> float:
        """
        Algorithm 2: CAATR trust-region update
        """
        if iteration < 2:
            return self.delta
            
        # Get teammate drift (line 5 in Algorithm 2)
        teammate_drift = 0.0
        for j in range(self.n_agents):
            if j != agent_id:
                teammate_drift += self.policy_drift_history[j, -1]
        
        # Adaptive epsilon (line 6)
        epsilon = max(self.epsilon_base, 
                     min(self.epsilon_max, teammate_drift / 10))
        
        # Compute adaptive trust region (line 7)
        delta_i = self.caatr_C / (teammate_drift + epsilon)
        
        return delta_i
    
    def solve_dual_problem(self, agent_id: int, batch: Dict, 
                          old_policy, delta_i: float) -> Tuple[float, float, float]:
        """
        Solve the dual problem from Theorem 1 (Equation 4-5).
        """
        actions = np.array(batch['actions'][agent_id])
        advantages = batch['advantages'][agent_id]
        
        def phi_lambda(s, a_i, lambda_val):
            """λ-regularized advantage function (Equation 5)"""
            # This is simplified - full implementation would compute over all a'_i
            return np.max([advantages[i] - lambda_val * np.abs(actions[i] - a_i) 
                          for i in range(len(actions))])
        
        def dual_objective(lambda_val):
            """Dual objective from Equation 4"""
            if lambda_val < 0:
                return np.inf
            phi_values = [phi_lambda(None, actions[i], lambda_val) 
                         for i in range(len(actions))]
            return lambda_val * delta_i + np.mean(phi_values)
        
        # Optimize dual variable
        result = minimize_scalar(dual_objective, bounds=(0, 100), method='bounded')
        optimal_lambda = result.x
        
        # Compute optimal policy update using optimal lambda
        # (Simplified - full implementation follows Corollary 1)
        new_mean = old_policy.mean  # Placeholder
        new_std = old_policy.std    # Placeholder
        
        return new_mean, new_std, optimal_lambda
    
    def train(self):
        """
        Main training loop following Algorithm 1.
        """
        print(f"Training W-MATRPO with {self.n_agents} agents")
        print(f"CAATR constant C = {self.caatr_C}")
        
        for iteration in range(self.n_iterations):
            # Collect trajectories (line 3)
            batch = self.collect_batch()
            
            # Update critics
            for critic in self.critics:
                critic.update(batch['rewards'])
            
            # Store old policies for drift calculation
            old_policies = copy.deepcopy(self.policies)
            
            # Random agent ordering (line 5)
            agent_order = np.random.permutation(self.n_agents)
            
            # Sequential updates (lines 7-14)
            for k, agent_id in enumerate(agent_order):
                # Compute adaptive trust region (line 4)
                delta_i = self.compute_caatr_radius(agent_id, iteration)
                
                # Solve dual problem (line 10)
                new_mean, new_std, lambda_star = self.solve_dual_problem(
                    agent_id, batch, old_policies[agent_id], delta_i)
                
                # Update policy (line 11)
                self.policies[agent_id].update(new_mean, new_std)
            
            # Store policy drift (line 16)
            drifts = []
            for i in range(self.n_agents):
                drift = self.policies[i].wasserstein_distance(old_policies[i])
                drifts.append(drift)
            self.policy_drift_history = np.column_stack(
                [self.policy_drift_history, drifts])
            
            # Log progress
            if iteration % 100 == 0:
                actions = [p.mean for p in self.policies]
                avg_reward = np.mean(batch['rewards'])
                print(f"Iteration {iteration}: Actions = {actions:.3f}, "
                      f"Reward = {avg_reward:.3f}")
    
    def collect_batch(self) -> Dict:
        """Collect batch of trajectories."""
        # Placeholder - implement full batch collection
        batch = {
            'actions': [[] for _ in range(self.n_agents)],
            'rewards': [],
            'advantages': [[] for _ in range(self.n_agents)]
        }
        return batch


# Run experiments for all agent configurations
if __name__ == "__main__":
    for n_agents in [2, 3, 5, 7, 9]:
        print(f"\n{'='*60}")
        print(f"Running W-MATRPO with {n_agents} agents")
        print(f"{'='*60}")
        
        env = DifferentialGameEnv(n_agents=n_agents)
        wmatrpo = WMATRPO(env, n_agents=n_agents)
        wmatrpo.train()
        
        # Report final distance from global optimum
        final_actions = np.array([p.mean for p in wmatrpo.policies])
        distance = np.linalg.norm(final_actions - env.global_optimum)
        print(f"Final distance from global optimum: {distance:.4f}")
