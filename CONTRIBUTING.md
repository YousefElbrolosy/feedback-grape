# Contributing to feedbackGRAPE

Thank you for your interest in contributing to feedbackGRAPE! We welcome bug reports, feature suggestions, documentation improvements, and code contributions.

---

## 🛠️ Setting Up Your Development Environment

1. **Fork the repository** and clone your fork:
   git clone https://github.com/YOUR_USERNAME/feedbackGRAPE.git

2. **Install development dependencies**:
   pip install -U -r requirements.txt
   pip install -U -r requirements_dev.txt

3. **(Optional) For GPU support**:
   pip install -U -r requirements_gpu.txt

4. **Install JAX for CUDA 12** (if using GPU):
   pip install -U "jax[cuda12]==0.5.2"

5. **(Optional) Install Pandoc for notebook rendering**:
   conda install conda-forge::pandoc

---

## 🧪 Running Tests

We use `pytest` for testing. Run:

    pytest

This will execute all unit tests and produce a coverage report.

---

## ✅ Code Quality

Before submitting any code, make sure it passes the following checks:

- **Type Checking**: Run
      ```mypy feedback_grape```

- **Linting**: Run
      ```ruff check```

- **Formatting**: Run
      ```ruff format```

- **Full Pre-push Check**: Run
      ```tox```

This runs all tests, type checks, and linting in isolated environments.

---

## ✍️ Making a Contribution

1. Create a new branch from `dev`:
   git checkout -b feature/my-feature

2. Make your changes and **include tests** and **docstrings**.

3. Run `tox` to ensure everything passes.

4. Push your branch and open a **Pull Request (PR) to the dev branch**.

5. Make sure your PR:
   - Has a clear title and description
   - References relevant issues (if any)
   - Passes all CI checks

---

## 📚 Documentation Contributions

- Docs are in the `docs/` folder and built using Sphinx.
- To preview documentation changes locally:

      cd docs
      make html
      # Open docs/build/html/index.html

---

## 💬 Need Help?

If you have any questions:
- Open an issue
- Start a discussion in the GitHub Discussions tab
- Contact maintainers via email if necessary

We appreciate your time and effort to make feedback-grape better!
