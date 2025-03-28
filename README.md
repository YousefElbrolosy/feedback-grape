# FeedBack-GRAPE
This is the main repository for the GRAPE-Juice package containing feedback GRAPE techniques.

## Installation
To install dependencies necessary for the package <br>
`pip install -r requirements.txt` <br>
To install dependencies necessary for testing, linting, formating, ... <br>
`pip install -r requirements_dev.txt`

## Testing
Simply type `pytest`. This would also generate a coverage report.

### checking for dynamically typed errors
Simply type `mypy src`. This would give you type checking errors if any.

### Before Commiting and Pushing
Simply type `tox`. This would test the code on different environments.
