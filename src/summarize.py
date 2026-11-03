import pandas as pd
import ollama
from tqdm import tqdm
import time
import tracemalloc
import os

MODELS_TO_TEST = ["llama3.2", "gemma3:4b", "qwen3:4b"]
INPUT_FILE = "../data/news_summary.csv"
OUTPUT_FILE = "../data/news_summary_multi_model_metrics.csv"

print(f"Učitavam prvih 200 redaka iz datoteke '{INPUT_FILE}'...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Datoteka {INPUT_FILE} nije pronađena u trenutnom direktoriju!")

df = pd.read_csv(INPUT_FILE, encoding='latin1', nrows=100)

for model in MODELS_TO_TEST:
    model_clean = model.replace(":", "_")
    
    summary_col = f"{model_clean}_summary"
    time_col = f"{model_clean}_time_sec"
    memory_col = f"{model_clean}_memory_mb"
    
    if summary_col not in df.columns:
        df[summary_col] = None
    else:
        df[summary_col] = df[summary_col].astype(object)
        
    if time_col in df.columns:
        df.drop(columns=[time_col], inplace=True)
    df[time_col] = pd.Series(dtype='float64')
        
    if memory_col in df.columns:
        df.drop(columns=[memory_col], inplace=True)
    df[memory_col] = pd.Series(dtype='float64')

for model in MODELS_TO_TEST:
    model_clean = model.replace(":", "_")
    summary_col = f"{model_clean}_summary"
    time_col = f"{model_clean}_time_sec"
    memory_col = f"{model_clean}_memory_mb"
    
    print(f"\n--- Pokrećem testiranje za model: {model} ---")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=model):
        
        if pd.notna(row[summary_col]) and row[summary_col] != "":
            continue

        article = str(row["ctext"])
        article = article[:5000]

        prompt = f"""
        You are an expert news editor.
        Summarize the following news article.

        Requirements:
        - Use exactly 3 sentences.
        - Keep only the most important information.
        - Use journalistic style.
        - Avoid repetition.

        Article:
        {article}
        """

        try:
            tracemalloc.start()
            start_time = time.time()

            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            execution_time = end_time - start_time
            peak_memory_mb = peak / (1024 * 1024)
            df.at[index, summary_col] = response['message']['content']
            df.at[index, time_col] = float(round(execution_time, 2))
            df.at[index, memory_col] = float(round(peak_memory_mb, 2))

        except Exception as e:
            print(f"\nGreška na indeksu {index} za model {model}: {e}")
            if tracemalloc.is_tracing():
                tracemalloc.stop()
                
            df.at[index, summary_col] = f"ERROR: {str(e)}"
            df.at[index, time_col] = 0.0
            df.at[index, memory_col] = 0.0
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Privremeni rezultati za model {model} spremljeni u '{OUTPUT_FILE}'.")
print(f"\nKonačni rezultati su u datoteci: '{OUTPUT_FILE}'")