# --- 0. INSTALL AND SETUP (Run these lines first if you haven't already) ---
# !pip install accelerate transformers pandas torch
# NOTE: The HuggingFace login needs to be run only once per session
from huggingface_hub import login
# login(token='use your token') # Replace with your actual token line

import pandas as pd
import json
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import sys
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress transformers warnings

# --- 1. LLAMA 3 INITIALIZATION (THIS MUST RUN FIRST) ---
try:
    # Use your actual login token line here
    # Assuming you have already run the login command successfully in your terminal/notebook
    print("Initializing Llama 3 pipeline...")
    
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe_tokenizer = pipe.tokenizer
    print("Llama 3 pipeline loaded successfully.")

except Exception as e:
    print(f"\nFATAL: Failed to load Llama 3 pipeline. Check your environment, token, and hardware.")
    print(f"Error details: {e}")
    sys.exit(1) # Exit the script if the model cannot be loaded


# --- 2. CONFIGURATION ---
# IMPORTANT: Use the exact path to your CSV file
file_path = '/home/gella.saikrishna/.cache/kagglehub/datasets/dataanalyst001/all-capital-cities-in-the-world/versions/1/all capital cities in the world.csv'
QUERY_COLUMN = 'Country' 
EVAL_COLUMNS = ['Capital City', 'Continent', 'Latitude', 'Longitude'] 


# --- 3. UPDATED LLAMA 3 PREDICTION FUNCTION ---
# This function now expects the 'pipe' and 'pipe_tokenizer' to be defined globally 
# by the initialization step above.
def get_llama3_prediction(country_name, pipe, pipe_tokenizer):
    """
    Takes a country name, queries the Llama 3 pipeline, and returns a dictionary.
    """
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

    terminators = [
        pipe_tokenizer.eos_token_id,
        pipe_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # Run Inference with deterministic settings
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
    )

    # Extract and Parse the JSON
    raw_output = outputs[0]["generated_text"][len(prompt):].strip()
    
    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    
    if json_match:
        json_string = json_match.group(0)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            # Handle cases where the JSON is invalid
            print(f"Warning: Failed to parse JSON for {country_name}. Raw output: {raw_output[:50]}...")
            return {col: "" for col in EVAL_COLUMNS}
    else:
        # Handle cases where no JSON is found
        # print(f"Warning: No valid JSON found for {country_name}. Raw output: {raw_output[:50]}...")
        return {col: "" for col in EVAL_COLUMNS}


# --- 4. EVALUATION LOGIC (Main loop) ---
def calculate_efficiency(df, query_col, eval_cols, pipe, pipe_tokenizer):
    """Loops through the dataset, gets Llama 3 predictions, and calculates efficiency."""
    
    for col in eval_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        
    correct_counts = {col: 0 for col in eval_cols}
    total_count = len(df)

    print(f"Starting evaluation of {total_count} countries. This may take some time...")
    
    for index, row in df.iterrows():
        country = row[query_col]
        
        # Pass the initialized pipe and tokenizer objects
        llm_response_dict = get_llama3_prediction(country, pipe, pipe_tokenizer)
        
        if (index + 1) % 10 == 0 or (index + 1) == total_count:
            print(f"Processed {index + 1}/{total_count} entries...")
        
        for col in eval_cols:
            true_value = row[col]
            predicted_value = str(llm_response_dict.get(col, '')).strip().lower()
            
            is_correct = (predicted_value == true_value)
            
            # Robust Comparison for Latitude/Longitude (Tolerance 0.05)
            if col in ['Latitude', 'Longitude']:
                true_num_str = re.sub(r'[^0-9.-]', '', true_value)
                pred_num_str = re.sub(r'[^0-9.-]', '', predicted_value)
                
                try:
                    true_num = float(true_num_str)
                    pred_num = float(pred_num_str)
                    
                    if abs(true_num - pred_num) < 0.05:
                         is_correct = True
                    else:
                        is_correct = False
                except ValueError:
                    is_correct = False

            if is_correct:
                correct_counts[col] += 1
        
    efficiency = {
        col: f"{correct_counts[col] / total_count * 100:.2f}%" 
        for col in eval_cols
    }
    
    return efficiency, total_count, correct_counts


# --- 5. EXECUTION ---
try:
    # Load the ground truth data
    data = pd.read_csv(file_path)
    
    # Run the evaluation, passing the pipe and tokenizer
    efficiency_results, total, correct = calculate_efficiency(data, QUERY_COLUMN, EVAL_COLUMNS, pipe, pipe_tokenizer)
    
    print("\n" + "="*50)
    print("🧠 Llama 3 Evaluation Results (Efficiency / Accuracy)")
    print("="*50)
    print(f"Total Countries Evaluated: {total}")
    print("\nAccuracy by Column:")
    
    results_table = pd.DataFrame([efficiency_results]).T
    results_table.columns = ['Efficiency']
    results_table.index.name = 'Column'
    
    print(results_table.to_markdown(numalign="left", stralign="left"))

    print("\nRaw Counts:")
    for col in EVAL_COLUMNS:
        print(f"- {col}: {correct[col]}/{total} correct")

except FileNotFoundError:
    print(f"\nFATAL ERROR: The file was not found at the configured path:\n{file_path}")
    print("Please ensure the path is correct.")
except Exception as e:
    # Catch any remaining errors outside of the loading and processing functions
    print(f"\nAn unhandled error occurred during execution: {e}")
