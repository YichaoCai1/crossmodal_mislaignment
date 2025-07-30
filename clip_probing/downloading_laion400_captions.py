import os
import requests
import pandas as pd
from tqdm import trange

# Base URL template
BASE_URL = "https://huggingface.co/api/datasets/laion/laion400m/parquet/default/train/{}.parquet"

# Directories
download_dir = "laion400m-metadata"
caption_dir = os.path.join(download_dir, "captions")
os.makedirs(download_dir, exist_ok=True)
os.makedirs(caption_dir, exist_ok=True)

for i in trange(128, desc="Processing LAION-400M"):
    try:
        file_name = f"{i}.parquet"
        parquet_url = BASE_URL.format(i)
        parquet_path = os.path.join(download_dir, file_name)
        caption_path = os.path.join(caption_dir, f"{i}.txt")

        # Step 1: Download
        print(f"\nDownloading {file_name} ...")
        with requests.get(parquet_url, stream=True) as r:
            r.raise_for_status()
            with open(parquet_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Step 2: Extract captions
        print(f"Extracting captions from {file_name} ...")
        df = pd.read_parquet(parquet_path, columns=["caption"])
        df["caption"].dropna().to_csv(caption_path, index=False, header=False, mode='a', encoding='utf-8')

        # Step 3: Delete parquet
        os.remove(parquet_path)
        print(f"Deleted {file_name}")

    except Exception as e:
        print(f"Error processing file {i}: {e}")
