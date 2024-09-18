import requests
import datasets
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, load_dataset

def get_subsets():
    pathlist = Path("/home/nus_cisco_wp1/Projects/Venus/venus_temp").glob('cpp-*')
    subsets = [path.name for path in pathlist]
    return subsets

subsets = get_subsets()
instances = list()
instance_ids = set()

print(f"[+] Loading old instances...")
ds = load_dataset("Elfsong/venus", "cpp")
for instance in ds['train'].to_list():
    instance_ids.add(instance['question_id'])
    instances.append(instance)
print(f"[+] [{len(instance_ids)}] old instances Loaded.")
    
print("[+] Loading new instances...")
for subset_name in tqdm(subsets):
    print(f"Current Subset [{subset_name}]")    
    try: 
        if "cpp-" in subset_name:
            ds = load_dataset("Elfsong/venus_temp", subset_name)
            for instance in ds['train'].to_list():
                if instance['question_id'] not in instance_ids:
                    instance_ids.add(instance['question_id'])
                    instances.append(instance)
    except datasets.exceptions.DatasetGenerationError as e:
        print(f"Empty Dataset {subset_name}")
    except Exception as e:
        print(f"{subset_name} Error: {e}")
print(f"[+] [{len(instance_ids)}] old instances Loaded.")
    
print("[+] Uploading the new dataset...")
ds = Dataset.from_pandas(pd.DataFrame(data=instances))
ds.push_to_hub("Elfsong/venus", "cpp")

print("[+] Checking the new dataset...")
ds = load_dataset("Elfsong/venus", "cpp")
print(ds['train'])

