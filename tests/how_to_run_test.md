# Run all tests
```bash
pytest
```

# Run with verbose output
```bash
pytest -v
```
# Run with coverage report
```bash
pytest --cov
```
# Run specific test file
```bash
pytest tests/test_example.py
```
# Run specific test function
```bash
pytest tests/test_example.py::test_basic_example
```

You should see output like:
```
collected 2 items

tests/test_example.py::test_basic_example PASSED
tests/test_example.py::test_string_operations PASSED

====== 2 passed in 0.02s ======
```
