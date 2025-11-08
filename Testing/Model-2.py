import pandas as pd
import json
import re
import torch
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import sys
import os

# --- Configuration & Setup ---

# Suppress warnings from the transformers library for a clean output
logging.getLogger("transformers").setLevel(logging.ERROR)

# IMPORTANT: Use the full, absolute path from your environment
file_path = '/home/gella.saikrishna/.cache/kagglehub/datasets/dataanalyst001/all-capital-cities-in-the-world/versions/1/all capital cities in the world.csv'
QUERY_COLUMN = 'Country' 
EVAL_COLUMNS = ['Capital City', 'Continent', 'Latitude', 'Longitude'] 
ROWS_TO_EVALUATE = 5 # As requested, we will evaluate only the first 5 entries

# --- 1. LLM Initialization (DeepSeek-R1-Distill-Qwen-1.5B) ---
try:
    print(" Initializing DeepSeek-R1-Distill-Qwen-1.5B pipeline...")
    
    # NOTE: You may need to run huggingface_hub.login() separately if this is your first time.
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        # Using torch.float16 and 'auto' device map for typical GPU setups
        torch_dtype=torch.float16, 
        device_map="auto",
    )
    pipe_tokenizer = pipe.tokenizer
    print(" DeepSeek pipeline loaded successfully.")

except Exception as e:
    print(f"\nFATAL: Failed to load DeepSeek pipeline. Check environment, token, and hardware.")
    print(f"Error details: {e}")
    # Setting pipe to None so the rest of the script handles the failure gracefully
    pipe = None 
    pipe_tokenizer = None


# --- 2. LLM Prediction Function ---
def get_llm_prediction(country_name, pipe, pipe_tokenizer):
    """
    Takes a country name, queries the LLM pipeline, and returns a dictionary 
    with the expected columns, or empty strings on failure.
    """
    if pipe is None:
        return {col: "LLM_ERROR" for col in EVAL_COLUMNS}
        
    prompt_instruction = f"""
    You are an expert geographical information system. 
    Your task is to provide the Capital City, Continent, Latitude, and Longitude for the requested country.
    You MUST respond ONLY with a valid JSON object. DO NOT include any text outside the JSON object.
    The JSON structure must be: {{"Capital City": "...", "Continent": "...", "Latitude": "...", "Longitude": "..."}}
    """
    
    messages = [
        {"role": "system", "content": prompt_instruction},
        {"role": "user", "content": f"Provide the geographical data for: {country_name}"},
    ]

    prompt = pipe_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Define terminators for Qwen-based models
    terminators = [pipe_tokenizer.eos_token_id]
    qwen_eot_id = pipe_tokenizer.convert_tokens_to_ids("<|im_end|>")
    if qwen_eot_id is not None:
        terminators.append(qwen_eot_id)
    
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0,
        )
        raw_output = outputs[0]["generated_text"][len(prompt):].strip()
        
        # Use regex to robustly find the JSON object in the output
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(0)
            return json.loads(json_string)
        else:
            print(f"Warning: No JSON found for {country_name}.")
            return {col: "" for col in EVAL_COLUMNS}
    except Exception as e:
        print(f"Error during LLM inference for {country_name}: {e}")
        return {col: "" for col in EVAL_COLUMNS}


# --- 3. Evaluation Logic ---
def calculate_accuracy(df, query_col, eval_cols, pipe, pipe_tokenizer):
    """Loops through the dataframe rows, gets LLM predictions, and calculates accuracy."""
    
    # Normalize ground truth data for comparison
    df_eval = df.head(ROWS_TO_EVALUATE).copy()
    for col in eval_cols:
        df_eval[col] = df_eval[col].astype(str).str.strip().str.lower()
        
    correct_counts = {col: 0 for col in eval_cols}
    total_count = len(df_eval)

    print(f"\n--- Starting Evaluation for First {total_count} Entries ---")
    
    for index, row in df_eval.iterrows():
        country = row[query_col]
        llm_response_dict = get_llm_prediction(country, pipe, pipe_tokenizer)
        
        print(f"\n[{index+1}/{total_count}] Country: {country}")
        
        for col in eval_cols:
            true_value = row[col]
            predicted_value = str(llm_response_dict.get(col, '')).strip().lower()
            
            is_correct = (predicted_value == true_value)
            
            # Special Robust Comparison for Latitude/Longitude (within a small tolerance)
            if col in ['Latitude', 'Longitude']:
                # Clean strings to extract numbers only
                true_num_str = re.sub(r'[^\d\.\-]', '', true_value.replace('n', '').replace('s', '').replace('e', '').replace('w', ''))
                pred_num_str = re.sub(r'[^\d\.\-]', '', predicted_value.replace('n', '').replace('s', '').replace('e', '').replace('w', ''))
                
                try:
                    true_num = float(true_num_str)
                    pred_num = float(pred_num_str)
                    
                    # Check if the absolute difference is less than 0.1 (a small tolerance)
                    if abs(true_num - pred_num) < 0.1:
                        is_correct = True
                    else:
                        is_correct = False
                except ValueError:
                    # If conversion to float fails, it's considered incorrect
                    is_correct = False

            if is_correct:
                correct_counts[col] += 1
                result_status = "CORRECT"
            else:
                result_status = "INCORRECT"
                
            print(f"  - {col}: {result_status} (True: {true_value}, Predicted: {predicted_value})")
            
    # Calculate final accuracy percentages
    efficiency = {
        col: f"{correct_counts[col] / total_count * 100:.2f}% ({correct_counts[col]}/{total_count})" 
        for col in eval_cols
    }
    
    return efficiency, total_count, correct_counts


# --- 4. Main Execution Block ---
if __name__ == '__main__':
    if pipe is None:
        print("\nCannot proceed with evaluation due to LLM initialization failure.")
        sys.exit(1)
        
    try:
        # Load the ground truth data
        data = pd.read_csv(file_path)
        
        # Display the first 5 entries (matching your provided image)
        print("\nFirst 5 entries of the dataset (Ground Truth):")
        # Ensure Sno is displayed as an integer for readability
        data_display = data.head(ROWS_TO_EVALUATE).copy()
        data_display['Sno'] = data_display['Sno'].astype(int)
        print(data_display.to_string(index=False))
        
        # Run the evaluation
        efficiency_results, total, correct = calculate_accuracy(data, QUERY_COLUMN, EVAL_COLUMNS, pipe, pipe_tokenizer)
        
        # Print final results
        print("\n" + "="*50)
        print("🧠 DeepSeek Evaluation Results (Accuracy for First 5 Entries)")
        print("="*50)
        print(f"Total Countries Evaluated: {total}")
        
        results_table = pd.DataFrame([efficiency_results]).T
        results_table.columns = ['Accuracy']
        results_table.index.name = 'Column'
        
        print("\nAccuracy by Column:")
        # Display the results in a formatted table
        print(results_table.to_markdown(numalign="left", stralign="left"))
        
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The file was not found at the configured path:\n{file_path}")
        print("Please ensure the path is correct.")
    except Exception as e:
        print(f"\nAn unhandled error occurred during execution: {e}")
