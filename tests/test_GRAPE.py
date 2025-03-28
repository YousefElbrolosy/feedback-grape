import pytest
from GRAPE.main import hello_tests
# if first test fails, the rest won't be tested
# def test_hello_test():
#     assert hello_tests(1) == 1
#     assert hello_tests(2) == 3

# here it tells you which passed and which did not
@pytest.mark.parametrize("x, y", [(1, 2), (2, 3), (3, 4)])
def test_hello_test2(x,y):
    assert hello_tests(x) == y

# Testing for expecting an exception
def test_invalid_input():
    with pytest.raises(TypeError):
        hello_tests("hello")

# Check documentation for pytest for more decorators

