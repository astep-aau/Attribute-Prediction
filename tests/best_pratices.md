## Naming Convention
```python
def test_<function>_<scenario>():
    """Clear description of what's being tested"""
```
Examples:
- test_user_login_with_invalid_password()
- test_calculate_discount_raises_error_for_negative_values()

## AAA Pattern (Arrange-Act-Assert)
```python
def test_calculate_total():
    # Arrange - Set up test data
    items = [10, 20, 30]

    # Act - Execute the function
    result = calculate_total(items)

    # Assert - Verify the result
    assert result == 60
```

## Use Fixtures for Setup
```python
# conftest.py
@pytest.fixture
def sample_user():
    return User(name="John", email="john@example.com")

# test_users.py
def test_user_email(sample_user):
    assert sample_user.email == "john@example.com"
```
## Parametrize Similar Tests
```python
@pytest.mark.parametrize("input,expected", [
    (5, 25),
    (0, 0),
    (-3, 9),
])
def test_square(input, expected):
    assert square(input) == expected
```
## Test Exceptions
```python
def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)
```
