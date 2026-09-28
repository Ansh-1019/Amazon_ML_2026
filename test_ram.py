import pandas as pd
import time
import os

print("Loading S2...")
s2 = pd.read_csv('data/raw/dataset/test/test_source2.tsv', sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'])
print(f"S2 loaded, shape {s2.shape}, RAM: {s2.memory_usage(deep=True).sum()/1e9:.2f} GB")

print("Loading S3...")
s3 = pd.read_csv('data/raw/dataset/test/test_source3.tsv', sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'])
print(f"S3 loaded, shape {s3.shape}, RAM: {s3.memory_usage(deep=True).sum()/1e9:.2f} GB")

print("Total RAM needed for S2 + S3:", (s2.memory_usage(deep=True).sum() + s3.memory_usage(deep=True).sum())/1e9, "GB")
