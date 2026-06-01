import json
import os
import inspect
import importlib
import sys

# Add src and sciona-matcher to python path
sys.path.insert(0, "/Users/conrad/personal/sciona-atoms-dl/src")
sys.path.insert(0, "/Users/conrad/personal/sciona-matcher")

def extract_module_atoms(module_name):
    cdg_path = f"/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms/{module_name}/cdg.json"
    if not os.path.exists(cdg_path):
        print(f"CDG file not found: {cdg_path}")
        return

    with open(cdg_path, 'r') as f:
        data = json.load(f)

    # Try to import the atoms module
    try:
        py_module = importlib.import_module(f"sciona.atoms.{module_name.replace('/', '.')}.atoms")
    except Exception as e:
        print(f"Could not import atoms: {e}")
        py_module = None

    # Also try atoms_torch if it exists
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
        print(f"\n========================================================================")
        print(f"ATOM: {node_id} (Opaque: {node.get('is_opaque', False)})")
        print(f"Description: {node.get('description')}")
        print(f"Signature: {node.get('type_signature')}")
        print(f"========================================================================")

        func = None
        if py_module and hasattr(py_module, node_id):
            func = getattr(py_module, node_id)
        elif py_module_torch and hasattr(py_module_torch, node_id):
            func = getattr(py_module_torch, node_id)

        if func:
            try:
                source = inspect.getsource(func)
                print(source)
            except Exception as e:
                print(f"Error getting source: {e}")
        else:
            print("Implementation NOT found in atoms.py or atoms_torch.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_source.py <module_relative_path>")
        print("e.g. python3 extract_source.py dl/adversarial")
    else:
        extract_module_atoms(sys.argv[1])
