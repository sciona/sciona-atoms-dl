import json
import glob
import os
import re

path_pattern = "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms/**/cdg.json"
files = glob.glob(path_pattern, recursive=True)

all_py_files = glob.glob("/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms/**/*.py", recursive=True)
py_contents = {}
for py_file in all_py_files:
    with open(py_file, 'r', errors='ignore') as f:
        py_contents[py_file] = f.read()

results = []

for file_path in sorted(files):
    rel_path = os.path.relpath(file_path, "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms")
    with open(file_path, 'r') as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    for node in nodes:
        status = node.get("status")
        if status == "decomposed":
            continue
        node_id = node.get("node_id")
        
        found_in = []
        pattern = re.compile(rf"\bdef\s+{node_id}\b")
        for py_file, content in py_contents.items():
            if pattern.search(content):
                found_in.append(os.path.relpath(py_file, "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms"))
        
        results.append({
            "cdg": rel_path,
            "node_id": node_id,
            "status": status,
            "is_opaque": node.get("is_opaque", False),
            "found_in": found_in
        })

out_path = "/Users/conrad/.gemini/antigravity-cli/brain/e403b54c-33d6-4694-8d85-b8fcae3746f5/scratch/atom_implementations.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

implemented = [r for r in results if r["found_in"]]
missing = [r for r in results if not r["found_in"]]

print(f"Total atoms: {len(results)}")
print(f"Implemented in Python: {len(implemented)}")
print(f"Missing in Python: {len(missing)}")

# Show missing categories
missing_by_cdg = {}
for r in missing:
    missing_by_cdg[r["cdg"]] = missing_by_cdg.get(r["cdg"], 0) + 1
print("Missing by CDG file:")
for k, v in missing_by_cdg.items():
    print(f"  {k}: {v}")
