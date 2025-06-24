# feedbackGRAPE
This is the main repository for the feedbackGRAPE package (under development), eventually offering:

- vectorized, GPU-enabled, differentiable simulations of driven dissipative quantum systems via jax
- efficient quantum optimal control (gradient-ascent pulse engineering, GRAPE)
- including feedback (using the newly developed feedbackGRAPE approach)

Think of parallelized, highly efficient qutip with feedback control.

## Installation
To install dependencies necessary for the package <br>
`pip install -U -r requirements.txt` <br>
To install dependencies necessary for testing, linting, formating, ... <br>
`pip install -U -r requirements_dev.txt` <br>
To be able to render the jupyter notebooks, make sure to install pandoc, in conda the command is: <br> 
`conda install conda-forge::pandoc` <br>
To be able to run your code on GPUs please make sure to install jax[cuda12] using the following command: <br>
`pip install -U "jax[cuda12]==0.5.2"` or <br>
`pip install -U -r requirements_gpu.txt` <br>


## Documentation
For Development: Enter the command `cd docs` then `make html` and then open index.html in docs/build/html with Live Server to take a look at the Documentaion.

## Testing
Simply type `pytest`. This would also generate a coverage report.

### checking for dynamically typed errors
Simply type `mypy feedback_grape`. This would give you type checking errors if any.

### linting and formating
For Linting `ruff check` <br>
For Formating `ruff format` <br>

### Before Commiting and Pushing
Simply type `tox`. This would test the code on different environments.

### References

FeedbackGRAPE was introduced in <a href="https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.030305">Porotti, Peano, Marquardt, PRX Quantum 4, 030305 (2023)</a>. It enables the addition of feedback to GRAPE-style numerical quantum optimal control, important for modern tasks like quantum state stabilization or quantum error correction. In addition, it is formulated in such a way that the use of neural networks and modern autodifferentiation frameworks is easy. See a first full-fledged application to the Gottesman-Kitaev-Preskill bosonic code quantum error correction in <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.134.020601">Puviani et al, Phys. Rev. Lett. 134, 020601 (2025)</a>. Both of these papers come with github repositories, but with code specialized to the examples and use cases: <a href="https://github.com/rporotti/feedback_grape">rporotti/feedback_grape</a> and <a href="https://github.com/Matteo-Puviani/GQF">Matteo-Puviani/GQF</a>. The purpose of the framework offered in this repository here is to make it very easy for everyone to do the same for their own problems, with minimum overhead.

