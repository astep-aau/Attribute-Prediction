"""Custom exceptions for the application"""

class NotFoundException(Exception):
    """Raised when a requested resource is not found"""
    pass

class InvalidUUIDException(Exception):
    """Raised when an invalid UUID is provided"""
    pass

class DatabaseException(Exception):
    """Raised when database operations fail"""
    pass

class InvalidTimeRangeException(Exception):
    """Raised when start_time is after end_time"""
    pass

class ValidationException(Exception):
    """Raised for custom validation errors"""
    pass

class ForeignKeyViolationException(Exception):
    """Raised when a foreign key constraint is violated"""
    pass

class FileOperationException(Exception):
    """Raised when file operations (CSV read/write) fail"""
    pass

class CSVParseException(Exception):
    """Raised when CSV parsing fails"""
    pass
