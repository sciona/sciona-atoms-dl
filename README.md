# sciona-atoms-dl

Deep learning provider repo for `sciona.atoms.*` namespace packages.

This repo contains atoms derived from deep learning competition solutions.
Atoms are organized by the level of framework coupling:

- **Framework-agnostic** atoms use only numpy/scipy and implement the
  mathematical operation independent of any DL framework. These have no
  torch/tensorflow dependency.
- **Framework-aware** atoms contain PyTorch or TensorFlow implementations.
  These are gated behind optional dependencies (`pip install sciona-atoms-dl[torch]`).

## Design principles

1. **Extract the math, not the framework.** Where possible, atoms implement
   the core algorithm in numpy. Framework-specific atoms exist only when
   the operation is inherently tied to autograd (loss functions that must
   be differentiable, custom layers that participate in backprop).

2. **Conceptual nodes for architectures.** Full neural network architectures
   (e.g., 3D U-Net) are represented as conceptual CDG nodes with
   `concept_type=neural_network` and `is_opaque=True`. The atom captures
   the architecture specification and forward-pass logic, not training
   infrastructure.

3. **Loss functions as first-class atoms.** Loss functions use
   `concept_type=loss_function` and connect to training nodes via
   `callable_injection` edges.

## Directory structure

```
src/sciona/atoms/dl/
├── detection/          # 3D object detection components
├── adversarial/        # Adversarial attack/defense operations
├── training/           # Training-loop primitives (OHEM, sampling, etc.)
└── loss/               # Custom loss functions
```
