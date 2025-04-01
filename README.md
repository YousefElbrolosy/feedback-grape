# feedbackGRAPE
This is the main repository for the feedbackGRAPE package (under development), eventually offering:

- vectorized, GPU-enabled, differentiable simulations of driven dissipative quantum systems via jax
- efficient quantum optimal control (gradient-ascent pulse engineering, GRAPE)
- including feedback (using the newly developed feedbackGRAPE approach)

Think of parallelized, highly efficient qutip with feedback control.

## Installation
To install dependencies necessary for the package <br>
`pip install -r requirements.txt` <br>
To install dependencies necessary for testing, linting, formating, ... <br>
`pip install -r requirements_dev.txt`

## Documentation
For Development: Enter the command `make html` and then open index.html with Live Server to take a look at the Documentaion

## Testing
Simply type `pytest`. This would also generate a coverage report.

### checking for dynamically typed errors
Simply type `mypy src`. This would give you type checking errors if any.

### linting and formating
For Linting `ruff check` <br>
For Formating `ruff format` <br>

### Before Commiting and Pushing
Simply type `tox`. This would test the code on different environments.

### References

FeedbackGRAPE was introduced in <a href="https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.030305">Porotti, Peano, Marquardt, PRX Quantum 4, 030305 (2023)</a>. It enables the addition of feedback to GRAPE-style numerical quantum optimal control, important for modern tasks like quantum state stabilization or quantum error correction. In addition, it is formulated in such a way that the use of neural networks and modern autodifferentiation frameworks is easy. See a first full-fledged application to the Gottesman-Kitaev-Preskill bosonic code quantum error correction in <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020601">Puviani et al, Phys. Rev. Lett. 134, 020601 (2025)</a>. Both of these papers come with github repositories, but with code specialized to the examples and use cases: <a href="https://github.com/rporotti/feedback_grape">rporotti/feedback_grape</a> and <a href="https://github.com/Matteo-Puviani/GQF">Matteo-Puviani/GQF</a>.

