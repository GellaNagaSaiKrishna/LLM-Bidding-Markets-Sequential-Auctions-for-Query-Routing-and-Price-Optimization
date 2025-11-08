import pandas as pd

# Use the full, absolute path where your kagglehub download is located
file_path = '/home/gella.saikrishna/.cache/kagglehub/datasets/dataanalyst001/all-capital-cities-in-the-world/versions/1/all capital cities in the world.csv'

try:
    # 1. Load the data into a DataFrame
    # Note the use of the full path, which should work in your local VS Code environment.
    df = pd.read_csv(file_path)
    
    # 2. Print the first 5 rows (and all columns)
    print("First 5 entries of the dataset:")
    print(df.head())
    
    # 3. Optional: Print the column names and data types
    print("\n--- Column Information ---")
    print(df.info())

except FileNotFoundError:
    print(f"Error: File not found at the specified path: {file_path}")
    print("Please double-check the path or ensure the file was successfully downloaded.")

except Exception as e:
    print(f"An error occurred: {e}")
