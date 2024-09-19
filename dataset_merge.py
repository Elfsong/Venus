# coding:utf-8

import datasets
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, load_dataset

  
parser = argparse.ArgumentParser()
parser.add_argument('--language', default="python3") 
args = parser.parse_args()

def get_subsets():
    pathlist = Path("/home/nus_cisco_wp1/Projects/venus_temp").glob(f'{args.language}-*')
    subsets = [path.name for path in pathlist]
    return subsets

subsets = get_subsets()
instances = list()
instance_ids = set()

print(f"🟢 Loading old instances...")
try:
    ds = load_dataset("Elfsong/venus", args.language)
    for instance in ds['train'].to_list():
        question_id = int(instance['question_id'])
        if question_id not in instance_ids:
            instance_ids.add(instance['question_id'])
            instances.append(instance)
    old_instance_count = len(instance_ids)
    print(f"[+] {old_instance_count} instances Loaded.")
    print("=====" * 5)
except ValueError as e:
    old_instance_count = 0
    print(f"[-] Empty dataset {args.language}, will create a new dataset.")
 
print(f"🟢 Loading new instances...")
for subset_name in tqdm(subsets):
    print(f"[+] Current Subset [{subset_name}]")    
    try: 
        if f"{args.language}-" in subset_name:
            ds = load_dataset("Elfsong/venus_temp", subset_name)
            for instance in ds['train'].to_list():
                if instance['question_id'] not in instance_ids:
                    instance_ids.add(instance['question_id'])
                    instances.append(instance)
    except datasets.exceptions.DatasetGenerationError as e:
        print(f"[-] Empty Dataset {subset_name}")
    except Exception as e:
        print(f"[-] {subset_name} Error: {e}")
new_instance_count = len(instance_ids)
print(f"[+] {new_instance_count} instances loaded. {new_instance_count-old_instance_count} instances added.")
print("=====" * 5)
    
print("🟢 Uploading the new dataset...")
df = pd.DataFrame(data=instances)
df['question_id'] = df['question_id'].astype('int64')
ds = Dataset.from_pandas(df)
ds.push_to_hub("Elfsong/venus", args.language)
print("=====" * 5)

print("🟢 Checking the new dataset...")
ds = load_dataset("Elfsong/venus", args.language)
print(ds['train'])
print("=====" * 5)

