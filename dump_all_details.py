import json
import glob
import os
import inspect
import importlib
import sys

sys.path.insert(0, "/Users/conrad/personal/sciona-atoms-dl/src")
sys.path.insert(0, "/Users/conrad/personal/sciona-matcher")

path_pattern = "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms/**/cdg.json"
files = glob.glob(path_pattern, recursive=True)

all_extracted = []

for file_path in sorted(files):
    rel_path = os.path.relpath(file_path, "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms")
    module_name = os.path.dirname(rel_path)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    try:
        py_module = importlib.import_module(f"sciona.atoms.{module_name.replace('/', '.')}.atoms")
    except Exception:
        py_module = None
        
    try:
        py_module_torch = importlib.import_module(f"sciona.atoms.{module_name.replace('/', '.')}.atoms_torch")
    except Exception:
        py_module_torch = None
        
    nodes = data.get("nodes", [])
    for node in nodes:
        status = node.get("status")
        if status == "decomposed":
            continue
            
        node_id = node.get("node_id")
        is_opaque = node.get("is_opaque", False)
        description = node.get("description", "")
        type_signature = node.get("type_signature", "")
        
        func = None
        if py_module and hasattr(py_module, node_id):
            func = getattr(py_module, node_id)
        elif py_module_torch and hasattr(py_module_torch, node_id):
            func = getattr(py_module_torch, node_id)
            
        code_snippet = ""
        docstring = ""
        if func:
            try:
                docstring = inspect.getdoc(func) or ""
                source = inspect.getsource(func)
                # Keep first 15 lines of source
                lines = source.splitlines()
                code_snippet = "\n".join(lines[:15])
            except Exception as e:
                code_snippet = f"Error getting source: {e}"
        else:
            code_snippet = "No Python implementation found"
            
        all_extracted.append({
            "cdg": rel_path,
            "node_id": node_id,
            "name": node.get("name"),
            "status": status,
            "is_opaque": is_opaque,
            "description": description,
            "type_signature": type_signature,
            "docstring": docstring,
            "code_snippet": code_snippet
        })

out_path = "/Users/conrad/.gemini/antigravity-cli/brain/e403b54c-33d6-4694-8d85-b8fcae3746f5/scratch/all_extracted_atoms.json"
with open(out_path, 'w') as f:
    json.dump(all_extracted, f, indent=2)

print(f"Successfully extracted {len(all_extracted)} atoms.")
