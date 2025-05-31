import os
import json

def is_valid_ipynb(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "cells" in data and "nbformat" in data
    except Exception as e:
        return False, str(e)

print("Checking notebooks in ./src...")
for root, _, files in os.walk("C:/Users/Steel/Desktop/Projects/intel-sweep-v2/intel-sweep/src"):
    for file in files:
        if file.endswith(".ipynb"):
            path = os.path.join(root, file)
            valid, error = is_valid_ipynb(path), ""
            if isinstance(valid, tuple):
                valid, error = valid
            if not valid:
                print(f"❌ Invalid notebook: {path}")
                if error:
                    print(f"    Error: {error}")
            else:
                print(f"✅ Valid notebook: {path}")