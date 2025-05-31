import os
import subprocess
import json

# Set your paths
source_dir = "./src"
target_base = "/Repos/ricciots@uwindsor.ca/intel-sweep"

# Allowed extensions
valid_extensions = {".scala", ".py", ".sql", ".SQL", ".r", ".R", ".ipynb", ".html", ".dbc"}

def is_valid_ipynb(file_path):
    if not file_path.endswith(".ipynb"):
        return True
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return "cells" in content and "nbformat" in content
    except Exception:
        return False

# Walk through the directory and import valid files
for root, _, files in os.walk(source_dir):
    for file in files:
        ext = os.path.splitext(file)[1]
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, source_dir)
        target_path = os.path.join(target_base, rel_path).replace("\\", "/")

        # Check extension
        if ext in valid_extensions and is_valid_ipynb(full_path):
            print(f"Importing: {full_path} -> {target_path}")
            subprocess.run([
                "databricks", "workspace", "import",
                full_path,
                target_path,
                "--overwrite"
            ])
        else:
            print(f"Skipping: {full_path}")