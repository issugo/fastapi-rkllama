from enum import Enum


class FieldType(Enum):
    """Enumeration of field types for configuration schema"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    PATH = "path"
