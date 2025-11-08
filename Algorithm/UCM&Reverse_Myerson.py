import pandas as pd
import numpy as np
import random
import math
import sys
from typing import List, Dict, Tuple

# --- Configuration Constants ---
C_UCB = 0.5
MODEL_CONFIG = {
    # Model-1: High cost/High reward prob (Simulated 3B)
    'Model-1': {'cost_per_token': 0.6, 'capacity': 3e9},
    # Model-2: Low cost/Low reward prob (Simulated 1B)
    'Model-2': {'cost_per_token': 0.1, 'capacity': 1e9}
}
REWARD_STRUCTURE = {
    # Assuming the total max reward of 70 in the final simulation output
    # corresponds to a scaled-up version of the original structure.
    # We will use the original structure for the logic and scale up the final result for presentation.
    'Capital City': 10, 'Continent': 10, 'Latitude': 25, 'Longitude': 25
}
MAX_REWARD = sum(REWARD_STRUCTURE.values()) # 70 points, matching the output
DF_FILE_NAME = '/home/gella.saikrishna/.cache/kagglehub/datasets/dataanalyst001/all-capital-cities-in-the-world/versions/1/all capital cities in the world.csv'
SIMULATION_ROUNDS = 200

# --- LLM Simulation Functions (using synthetic logic) ---

def simulate_llm_tokens_and_performance(country: str, model_name: str) -> Tuple[float, float, int]:
    """
    Simulates token usage, cost calculation, and reward for a single country query.
    
    Returns: (total_reward, total_cost, total_tokens)
    """
    
    # 1. Simulate Token Usage
    input_tokens = 50 
    country_length_factor = len(country) // 3
    output_tokens_base = 40 + country_length_factor
    output_tokens_variation = 5

    # Determine verbosity based on model size (FIXED: Numerical comparison)
    if MODEL_CONFIG[model_name]['capacity'] > 1e9: 
        # Larger model (Model-1: 3e9) might be slightly more verbose
        output_tokens = output_tokens_base + random.randint(0, output_tokens_variation)
    else:
        # Smaller model (Model-2: 1e9) might be slightly less verbose/more concise
        output_tokens = output_tokens_base + random.randint(-output_tokens_variation, 0)

    output_tokens = max(1, output_tokens)
    total_tokens = input_tokens + output_tokens

    # 2. Calculate Total Cost
    cost_per_token = MODEL_CONFIG[model_name]['cost_per_token']
    total_cost = total_tokens * cost_per_token

    # 3. Simulate Performance/Reward
    # Model-1 is better (90% base correctness), Model-2 is worse (70% base correctness)
    correctness_base = 0.9 if model_name == 'Model-1' else 0.7 
    total_reward = 0.0
    
    for _, reward_points in REWARD_STRUCTURE.items():
        # Introduce randomness around the base correctness probability
        correctness_prob = correctness_base + random.uniform(-0.1, 0.05)
        # Binary outcome: LLM is either correct (1) or incorrect (0)
        is_correct = 1 if random.random() < correctness_prob else 0
        total_reward += is_correct * reward_points
        
    return total_reward, total_cost, total_tokens

# --- UCB and Reverse Myerson Implementation ---

class LLMBanditSelector:
    """Implements UCB and Reverse Myerson's Virtual Valuation."""
    def __init__(self, models: List[str], c_ucb: float):
        self.models = models
        self.K = len(models)
        self.c_ucb = c_ucb
        self.t = 0
        
        # UCB State
        self.N_a = {model: 0 for model in models}
        self.Q_a = {model: 0.0 for model in models}
        
        # Reverse Myerson State
        self.a_history = {model: [] for model in models} 
        
    def _compute_ucb_index(self, model: str) -> float:
        """Calculates the UCB index for a given model (Q_t(a) + exploration_term)."""
        if self.N_a[model] == 0:
            return float('inf')
        
        # UCB Exploration Term: c * sqrt(ln(t) / N_a)
        exploration_term = self.c_ucb * math.sqrt(math.log(self.t) / self.N_a[model])
        ucb_index = self.Q_a[model] + exploration_term
        return ucb_index

    def _get_empirical_pdf_cdf(self, a_values: List[float], current_a: float, bin_count: int = 20) -> Tuple[float, float]:
        """Calculates empirical PDF and CDF values for the current 'a' based on history."""
        if not a_values:
            return 1.0, 0.5 

        a_all = np.array(a_values + [current_a])
        min_a, max_a = a_all.min(), a_all.max()
        
        if min_a == max_a:
            return 1.0, 1.0 if current_a >= min_a else 0.0
            
        bins = np.linspace(min_a, max_a, bin_count + 1)
        
        counts, bin_edges = np.histogram(a_values, bins=bins, density=False)
        total_samples = len(a_values)
        bin_width = bin_edges[1] - bin_edges[0]
        
        current_bin_index = np.digitize(current_a, bin_edges) - 1
        current_bin_index = np.clip(current_bin_index, 0, bin_count - 1)
        
        # Empirical PDF (density approximation)
        pdf_val = (counts[current_bin_index] / total_samples) / bin_width
        
        # Empirical CDF
        cdf_val = np.sum(counts[:current_bin_index + 1]) / total_samples
        
        pdf_val = max(pdf_val, 1e-6) # Prevent division by zero
        
        return pdf_val, cdf_val

    def _compute_virtual_valuation(self, a: float, model: str) -> float:
        """Reverse Myerson Virtual Valuation: a + (CDF(a) / PDF(a))"""
        a_history = self.a_history[model]
        pdf_a, cdf_a = self._get_empirical_pdf_cdf(a_history, a)
        
        # This formula is used for the valuation based on the bid 'a'
        virtual_valuation = a + (cdf_a / pdf_a)
        return virtual_valuation

    def select_model(self, country: str) -> str:
        """Selects the optimal model."""
        self.t += 1
        
        # Initial exploration: Ensure every arm is pulled once
        for model in self.models:
            if self.N_a[model] == 0:
                return model

        virtual_valuations: Dict[str, float] = {}
        
        for model in self.models:
            # 1. Simulate the hypothetical outcome (Reward, Cost)
            reward, cost, _ = simulate_llm_tokens_and_performance(country, model) 
            
            # 2. Calculate UCB Index and Exploration Term
            ucb_index = self._compute_ucb_index(model)
            exploration_term = ucb_index - self.Q_a[model] 
            
            # 3. Calculate 'a' value: a = Reward + UCB_Exploration_Term - Cost
            a = reward + exploration_term - cost
            
            # 4. Compute Reverse Myerson Virtual Valuation
            virtual_valuation = self._compute_virtual_valuation(a, model)
            virtual_valuations[model] = virtual_valuation
            
        # 5. Select the model with the LOWEST Virtual Valuation
        selected_model = min(virtual_valuations, key=virtual_valuations.get)
        
        return selected_model

    def update_model_stats(self, selected_model: str, reward: float, cost: float):
        """Updates the UCB and Reverse Myerson's history for the selected model."""
        
        # 1. UCB Update
        self.N_a[selected_model] += 1
        
        # Update empirical mean Q_t(a) (Incremental update formula)
        n = self.N_a[selected_model]
        old_q = self.Q_a[selected_model]
        new_q = old_q + (1 / n) * (reward - old_q)
        self.Q_a[selected_model] = new_q
        
        # 2. Calculate and update 'a' history for Myerson
        exploration_term = self.c_ucb * math.sqrt(math.log(self.t) / self.N_a[selected_model])
        a = reward + exploration_term - cost
        self.a_history[selected_model].append(a)


# --- Simulation Loop ---

def run_simulation(num_rounds: int, df: pd.DataFrame, selector: LLMBanditSelector):
    """Runs the multi-armed bandit simulation."""
    total_revenue = 0.0
    total_cost = 0.0
    
    COUNTRY_LIST = df['Country'].tolist()

    print(f"--- Starting LLM Selection Simulation ---")
    print(f"Models: {selector.models}, UCB-C: {selector.c_ucb}, Rounds: {num_rounds}")
    print(f"Max Reward per round: {MAX_REWARD}\n")

    for i in range(1, num_rounds + 1):
        # 1. Choose a random country for the query
        country = random.choice(COUNTRY_LIST)
        
        # 2. Select the optimal model
        selected_model = selector.select_model(country)
        
        # 3. Simulate the performance (this is the actual *observation*)
        reward, cost, tokens = simulate_llm_tokens_and_performance(country, selected_model) 
        
        # 4. Update the selector state
        selector.update_model_stats(selected_model, reward, cost)

        total_revenue += reward
        total_cost += cost
        net_value = reward - cost
        
        if i % 20 == 0 or i == num_rounds:
            print(f"Round {i}: Model={selected_model}, R={reward:.2f}, C={cost:.2f}, Net={net_value:.2f}")

    print("\n--- Simulation Complete ---")
    print(f"Total Rounds: {num_rounds}")
    print(f"Total Revenue: {total_revenue:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Net Profit (Revenue - Cost): {total_revenue - total_cost:.2f}")
    
    print("\n--- Model Pull Counts ---")
    for model, count in selector.N_a.items():
        print(f"{model}: {count} pulls")
        
    print("\n--- Final UCB Mean Rewards (Q_a) ---")
    for model, q_a in selector.Q_a.items():
        print(f"{model}: {q_a:.4f} average reward")
        
# --- Execution ---

if __name__ == '__main__':
    try:
        # 1. Load the synthetic dataset
        df_synthetic = pd.read_csv(DF_FILE_NAME)
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset not found at {DF_FILE_NAME}")
        sys.exit(1)

    # 2. Initialize the LLM Selector
    model_names = list(MODEL_CONFIG.keys())
    bandit_selector = LLMBanditSelector(models=model_names, c_ucb=C_UCB)

    # 3. Run the simulation
    run_simulation(SIMULATION_ROUNDS, df_synthetic, bandit_selector)
