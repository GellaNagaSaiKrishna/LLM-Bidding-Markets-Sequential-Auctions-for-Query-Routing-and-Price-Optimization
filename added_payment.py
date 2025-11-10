import pandas as pd
import numpy as np
import random
import math
import sys
from typing import List, Dict, Tuple, Optional

# --- Configuration Constants ---
C_UCB = 0.5
MAX_PAYMENT_CAP = 100.0 # Introduced a maximum cost the user is willing to pay
MODEL_CONFIG = {
    # Model-1: High cost/High reward prob (Simulated 3B)
    'Model-1': {'cost_per_token': 0.6, 'capacity': 3e9},
    # Model-2: Low cost/Low reward prob (Simulated 1B)
    'Model-2': {'cost_per_token': 0.1, 'capacity': 1e9}
}
REWARD_STRUCTURE = {
    'Capital City': 10, 'Continent': 10, 'Latitude': 25, 'Longitude': 25
}
MAX_REWARD = sum(REWARD_STRUCTURE.values()) # 70 points
DF_FILE_NAME = '/home/gella.saikrishna/.cache/kagglehub/datasets/dataanalyst001/all-capital-cities-in-the-world/versions/1/all capital cities in the world.csv'
SIMULATION_ROUNDS = 200

# --- LLM Simulation Functions (using synthetic logic) ---

def simulate_llm_tokens_and_performance(country: str, model_name: str) -> Tuple[float, float, int]:
    """
    Simulates token usage, cost calculation (bid), and reward.
    
    Returns: (total_reward, total_cost (BID), total_tokens)
    """
    
    # 1. Simulate Token Usage
    input_tokens = 50 
    country_length_factor = len(country) // 3
    output_tokens_base = 40 + country_length_factor
    output_tokens_variation = 5

    if MODEL_CONFIG[model_name]['capacity'] > 1e9: 
        output_tokens = output_tokens_base + random.randint(0, output_tokens_variation)
    else:
        output_tokens = output_tokens_base + random.randint(-output_tokens_variation, 0)

    output_tokens = max(1, output_tokens)
    total_tokens = input_tokens + output_tokens

    # 2. Calculate Total Cost (The Model's BID)
    cost_per_token = MODEL_CONFIG[model_name]['cost_per_token']
    bid_cost = total_tokens * cost_per_token # This is the model's bid
    
    # 3. Simulate Performance/Reward
    correctness_base = 0.9 if model_name == 'Model-1' else 0.7 
    total_reward = 0.0
    
    for _, reward_points in REWARD_STRUCTURE.items():
        correctness_prob = correctness_base + random.uniform(-0.1, 0.05)
        is_correct = 1 if random.random() < correctness_prob else 0
        total_reward += is_correct * reward_points
        
    return total_reward, bid_cost, total_tokens

# --- UCB and Reverse Myerson Implementation ---
class LLMBanditSelector:
    """Implements UCB and Reverse Myerson's Virtual Valuation."""
    def __init__(self, models: List[str], c_ucb: float):
        self.models = models
        self.K = len(models)
        self.c_ucb = c_ucb
        self.t = 0
        self.N_a = {model: 0 for model in models}
        self.Q_a = {model: 0.0 for model in models}
        self.a_history = {model: [] for model in models} 
        
    def _compute_ucb_index(self, model: str) -> float:
        if self.N_a[model] == 0:
            return float('inf')
        exploration_term = self.c_ucb * math.sqrt(math.log(self.t) / self.N_a[model])
        return self.Q_a[model] + exploration_term

    def _get_empirical_pdf_cdf(self, a_values: List[float], current_a: float, bin_count: int = 20) -> Tuple[float, float]:
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
        pdf_val = (counts[current_bin_index] / total_samples) / bin_width
        cdf_val = np.sum(counts[:current_bin_index + 1]) / total_samples
        pdf_val = max(pdf_val, 1e-6) 
        return pdf_val, cdf_val

    def _compute_virtual_valuation(self, a: float, model: str) -> float:
        a_history = self.a_history[model]
        pdf_a, cdf_a = self._get_empirical_pdf_cdf(a_history, a)
        return a + (cdf_a / pdf_a)

    def select_model(self, country: str) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Selects the optimal model. Returns (selected_model, all_simulated_bids).
        Returns (None, all_simulated_bids) if no model is selected.
        """
        self.t += 1
        
        virtual_valuations: Dict[str, float] = {}
        all_simulated_bids: Dict[str, float] = {}

        for model in self.models:
            # 1. Simulate the hypothetical outcome (Reward, Cost/BID)
            reward, bid_cost, _ = simulate_llm_tokens_and_performance(country, model) 
            all_simulated_bids[model] = bid_cost # Store the bid
            
            # Initial exploration check (must happen after bids are simulated)
            if self.N_a[model] == 0:
                 # In a bandit context, we must explore, but we need to ensure 
                 # the bid is under the MAX_PAYMENT_CAP to avoid running an unaffordable model initially.
                 if bid_cost <= MAX_PAYMENT_CAP:
                    return model, all_simulated_bids
                 else:
                    # If the required bid for an un-pulled arm is too high, 
                    # we continue to the next un-pulled arm or use the UCB logic.
                    continue
            
            # 2. Calculate UCB Index and Exploration Term
            ucb_index = self._compute_ucb_index(model)
            exploration_term = ucb_index - self.Q_a[model] 

            # 3. Calculate 'a' value: a = Reward + UCB_Exploration_Term - Cost (BID)
            a = reward + exploration_term - bid_cost
            
            # 4. Compute Reverse Myerson Virtual Valuation
            virtual_valuation = self._compute_virtual_valuation(a, model)
            virtual_valuations[model] = virtual_valuation
            
        if not virtual_valuations:
            # No models selected (happens if all un-pulled models bid too high)
            return None, all_simulated_bids

        # 5. Select the model with the LOWEST Virtual Valuation
        selected_model = min(virtual_valuations, key=virtual_valuations.get)
        
        # 6. FINAL CHECK: Check if the lowest bidder (the selected model) 
        # is affordable based on the MAX_PAYMENT_CAP rule.
        
        # NOTE: The selected model is the one that maximizes expected utility, 
        # but the selection must be invalidated if the required payment exceeds the cap.
        
        # Use the auction rule to determine the *required payment* for the selected model
        actual_payment = calculate_payment(selected_model, all_simulated_bids)
        
        # If the actual payment required exceeds the cap, or the model's own bid 
        # is greater than the cap, we select no LLM.
        if all_simulated_bids[selected_model] > MAX_PAYMENT_CAP or actual_payment > MAX_PAYMENT_CAP:
            # If the best choice is still too expensive, abort selection.
            return None, all_simulated_bids

        return selected_model, all_simulated_bids

    def update_model_stats(self, selected_model: str, reward: float, actual_payment: float, bid_cost: float):
        """Updates the UCB and Reverse Myerson's history for the selected model."""
        
        self.N_a[selected_model] += 1
        
        # UCB mean reward Q_a is updated based on the Net Value (Reward - Actual Payment)
        net_value_observed = reward - actual_payment
        
        n = self.N_a[selected_model]
        old_q = self.Q_a[selected_model]
        new_q = old_q + (1 / n) * (net_value_observed - old_q)
        self.Q_a[selected_model] = new_q
        
        # 'a' history update
        exploration_term = self.c_ucb * math.sqrt(math.log(self.t) / self.N_a[selected_model])
        a = reward + exploration_term - actual_payment
        self.a_history[selected_model].append(a)

# --- Payment Calculation ---

def calculate_payment(selected_model: str, all_bids: Dict[str, float]) -> float:
    """
    Calculates the actual payment based on min(MAX_PAYMENT_CAP, 2nd lowest bid).
    
    This function assumes a selection has ALREADY occurred and is used to 
    determine the final transaction cost.
    """
    
    bids = sorted(all_bids.values())
    
    # 1. Determine the 2nd lowest bid
    if len(bids) < 2:
        # In a 1-model scenario, the payment would typically be the min(Cap, 1st bid)
        second_lowest_bid = bids[0] 
    else:
        # Use the second lowest bid (index 1)
        second_lowest_bid = bids[1]

    # 2. The winning model's payment is min(Upper Bound, 2nd Lowest Bid)
    actual_payment = min(MAX_PAYMENT_CAP, second_lowest_bid)
    
    return actual_payment

# --- Simulation Loop ---

def run_simulation(num_rounds: int, df: pd.DataFrame, selector: LLMBanditSelector):
    """Runs the multi-armed bandit simulation with a second-price payment rule."""
    global MAX_PAYMENT_CAP 
    
    total_revenue = 0.0
    total_cost = 0.0 # Tracks total actual payments
    rounds_missed = 0
    
    COUNTRY_LIST = df['Country'].tolist()

    print(f"--- Starting LLM Selection Simulation (Second-Price Payment) ---")
    print(f"Models: {selector.models}, UCB-C: {selector.c_ucb}, Rounds: {num_rounds}")
    print(f"Max Reward per round: {MAX_REWARD}, Max Payment Cap: {MAX_PAYMENT_CAP}\n")

    for i in range(1, num_rounds + 1):
        country = random.choice(COUNTRY_LIST)
        
        # 1. Select the optimal model and get ALL simulated bids
        # Returns selected_model (str or None) and all_simulated_bids
        selected_model, all_simulated_bids = selector.select_model(country)
        
        if selected_model is None:
            # --- NO LLM SELECTED ---
            rounds_missed += 1
            if i % 20 == 0 or i == num_rounds:
                 print(f"Round {i}: NO LLM SELECTED (Bids: {all_simulated_bids['Model-1']:.2f}, {all_simulated_bids['Model-2']:.2f})")
            continue

        # 2. Simulate the performance observation using the model's original bid_cost
        reward, bid_cost, tokens = simulate_llm_tokens_and_performance(country, selected_model)
        
        # 3. Calculate the actual payment based on the auction rule
        actual_payment = calculate_payment(selected_model, all_simulated_bids)
        
        # 4. Update the selector state (using the actual_payment)
        selector.update_model_stats(selected_model, reward, actual_payment, bid_cost)

        total_revenue += reward
        total_cost += actual_payment 
        net_value = reward - actual_payment
        
        if i % 20 == 0 or i == num_rounds:
            print(f"Round {i}: Model={selected_model}, R={reward:.2f}, Bid={bid_cost:.2f}, Paid={actual_payment:.2f}, Net={net_value:.2f}")

    print("\n--- Simulation Complete ---")
    print(f"Total Rounds: {num_rounds}")
    print(f"Rounds Missed (Bid > Cap): {rounds_missed}")
    print(f"Total Revenue: {total_revenue:.2f}")
    print(f"Total Cost (Actual Payments): {total_cost:.2f}")
    print(f"Net Profit (Revenue - Cost): {total_revenue - total_cost:.2f}")
    
    print("\n--- Model Pull Counts ---")
    for model, count in selector.N_a.items():
        print(f"{model}: {count} pulls")
        
    print("\n--- Final UCB Mean Net Values (Q_a) ---")
    print("(Based on Reward - Actual Payment)")
    for model, q_a in selector.Q_a.items():
        print(f"{model}: {q_a:.4f} average net value")
        
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
