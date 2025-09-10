from enum import Enum


class FieldType(str, Enum):
    """Enumeration of field types for configuration schema"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    PATH = "path"
