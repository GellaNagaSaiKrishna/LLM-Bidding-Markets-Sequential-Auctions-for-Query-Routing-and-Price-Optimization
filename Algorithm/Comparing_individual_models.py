# Assuming all constants (MODEL_CONFIG, REWARD_STRUCTURE, SIMULATION_ROUNDS,
# DF_FILE_NAME, MAX_REWARD) and functions (simulate_llm_tokens_and_performance)
# are already defined from the previous cell.

def run_exclusive_model_simulation(num_rounds: int, df: pd.DataFrame, model_name: str):
    """
    Runs a simulation for a fixed number of rounds using only the specified model
    to calculate the total net profit.
    """
    total_revenue = 0.0
    total_cost = 0.0
    
    COUNTRY_LIST = df['Country'].tolist()

    print(f"--- Starting Exclusive Simulation: {model_name} ({num_rounds} Rounds) ---")
    
    for i in range(1, num_rounds + 1):
        # 1. Choose a random country for the query
        country = random.choice(COUNTRY_LIST)
        
        # 2. Simulate the performance (This is the actual *observation*)
        # The selected model is always the fixed model_name
        reward, cost, tokens = simulate_llm_tokens_and_performance(country, model_name) 
        
        total_revenue += reward
        total_cost += cost
        
        if i % 50 == 0 or i == num_rounds:
            net_value = reward - cost
            print(f"Round {i}: Model={model_name}, R={reward:.2f}, C={cost:.2f}, Net={net_value:.2f}")

    net_profit = total_revenue - total_cost
    
    print("\n--- Exclusive Simulation Complete ---")
    print(f"Model Used: {model_name}")
    print(f"Total Rounds: {num_rounds}")
    print(f"Total Revenue: {total_revenue:.2f}")
    print(f"Total Cost: {total_cost:.2f}")
    print(f"Net Profit (Revenue - Cost): {net_profit:.2f}")
    
    return net_profit

# --- Execution for Model-1 ---

if __name__ == '__main__':
    try:
        # Load the synthetic dataset (assuming DF_FILE_NAME is globally accessible)
        df_synthetic = pd.read_csv(DF_FILE_NAME)
    except NameError:
        print("Error: DF_FILE_NAME is not defined. Ensure you run the previous cell first.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset not found at {DF_FILE_NAME}")
        sys.exit(1)

    # Run simulation using only Model-1
    print("\n" + "="*60)
    profit_model_1 = run_exclusive_model_simulation(SIMULATION_ROUNDS, df_synthetic, 'Model-1')
    print("="*60)

    # Optional: Run simulation using only Model-2 for comparison
    print("\n" + "="*60)
    profit_model_2 = run_exclusive_model_simulation(SIMULATION_ROUNDS, df_synthetic, 'Model-2')
    print("="*60)
    
    print("\n--- Final Comparative Profit ---")
    print(f"Model-1 Exclusive Profit: {profit_model_1:.2f}")
    print(f"Model-2 Exclusive Profit: {profit_model_2:.2f}")
