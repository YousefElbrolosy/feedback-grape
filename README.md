# feedback-grape

**feedback-grape** is a high-performance Python package for simulating and optimizing quantum systems with feedback. It builds on the GRAPE (Gradient Ascent Pulse Engineering) method and introduces a new approach that integrates feedback in a natural, differentiable, and efficient way.

---

## 🚀 Features

- ✅ Vectorized, GPU-enabled, and differentiable simulations using JAX  
- ✅ Efficient quantum optimal control via GRAPE  
- ✅ Feedback support with the newly developed feedbackGRAPE technique  
- ✅ Think of it as a parallelized, feedback-enabled, high-performance alternative to QuTiP

---

## 📦 Installation

### For Users

Install the latest version via pip:
```
pip install feedback-grape
```
To enable GPU acceleration (CUDA 12):
```
pip install feedback-grape[cuda12]
```
### For Developers

Install development dependencies:
```
pip install -U -r requirements.txt
```
To include tools for testing, linting, and formatting:
```
pip install -U -r requirements_dev.txt
```
To run notebooks with proper rendering:
```
conda install conda-forge::pandoc
```
To use GPU support with JAX:
```
pip install -U -r requirements_gpu.txt
```
---

## 📚 Documentation

- User Guide: https://feedbackgrape.readthedocs.io
- Developer Docs:
```
cd docs  
make html
``` 
# Open docs/build/html/index.html using a Live Server or your browser

---

## ✅ Testing & Code Quality

### Run Tests

```pytest```

Generates test results and a coverage report.

```tox```

This tests the code across multiple python environments and ensures consistency.

### Type Checking

mypy feedback_grape

### Linting & Formatting

- Lint code:
```
ruff check
```
- Auto-format:
```
ruff format
```
---

## 📖 References

The feedbackGRAPE method was introduced in:

- Porotti, Peano, Marquardt, PRX Quantum 4, 030305 (2023)  
  [https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.030305](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.030305)

It extends traditional GRAPE by incorporating feedback control, which is crucial for applications like quantum state stabilization and quantum error correction. It is designed to work seamlessly with neural networks and automatic differentiation frameworks.

A full application to quantum error correction with the GKP code is shown in:

- Puviani et al., Phys. Rev. Lett. 134, 020601 (2025)  
  [https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020601](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020601)

---

## 🧠 Related Repositories

- https://github.com/rporotti/feedback_grape — Repository for feedbackGRAPE paper  
- https://github.com/Matteo-Puviani/GQF — Repository for GKP quantum error correction implementation