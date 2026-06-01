import json
import glob
import os

path_pattern = "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms/**/cdg.json"
files = glob.glob(path_pattern, recursive=True)

all_atoms = []

for file_path in sorted(files):
    rel_path = os.path.relpath(file_path, "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms")
    with open(file_path, 'r') as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    for node in nodes:
        status = node.get("status")
        if status != "decomposed":
            all_atoms.append({
                "file": rel_path,
                "node_id": node.get("node_id"),
                "name": node.get("name"),
                "status": status,
                "is_opaque": node.get("is_opaque", False),
                "type_signature": node.get("type_signature", "")
            })

out_path = "/Users/conrad/.gemini/antigravity-cli/brain/e403b54c-33d6-4694-8d85-b8fcae3746f5/scratch/all_atoms.txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    f.write(f"Total non-decomposed atoms found: {len(all_atoms)}\n")
    for idx, atom in enumerate(all_atoms):
        f.write(f"{idx+1}. [{atom['file']}] ID: {atom['node_id']} | Status: {atom['status']} | Is Opaque: {atom['is_opaque']}\n")

print(f"Wrote {len(all_atoms)} atoms to {out_path}")
