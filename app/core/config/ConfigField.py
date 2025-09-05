from typing import Generic, Optional, Union, List, Any

from core.config import T, FieldType


class ConfigField(Generic[T]):
    """Definition of a configuration field with type information and validation"""

    def __init__(
        self,
        field_type: FieldType,
        default: T,
        description: str = "",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        options: Optional[List[Any]] = None,
        item_type: Optional[FieldType] = None,
        required: bool = False,
    ):
        self.field_type = field_type
        self.default = default
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.options = options
        self.item_type = item_type
        self.required = required

    def validate(self, value: Any) -> T:
        """Validate a value against this field definition"""
        if value is None:
            if self.required:
                raise ValueError("Field is required but value is None")
            return self.default

        # Type conversion based on field_type
        converted_value = self._convert_value(value)

        # Range validation for numeric types
        if self.field_type in [FieldType.INTEGER, FieldType.FLOAT]:
            if self.min_value is not None and converted_value < self.min_value:
                raise ValueError(
                    f"Value {converted_value} is less than minimum {self.min_value}"
                )
            if self.max_value is not None and converted_value > self.max_value:
                raise ValueError(
                    f"Value {converted_value} is greater than maximum {self.max_value}"
                )

        # Options validation
        if self.options is not None and converted_value not in self.options:
            raise ValueError(
                f"Value {converted_value} is not in allowed options: {self.options}"
            )

        return converted_value

    def _convert_value(self, value: Any) -> T:
        """Convert a value to the appropriate type based on field_type"""
        try:
            if self.field_type == FieldType.STRING:
                return str(value)
            elif self.field_type == FieldType.INTEGER:
                if isinstance(value, str):
                    return int(value)
                return int(value)
            elif self.field_type == FieldType.FLOAT:
                if isinstance(value, str):
                    return float(value)
                return float(value)
            elif self.field_type == FieldType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "on", "y")
                return bool(value)
            elif self.field_type == FieldType.LIST:
                if isinstance(value, str):
                    items = [item.strip() for item in value.split(",") if item.strip()]
                    if self.item_type:
                        # Convert each item to the specified type
                        temp_field = ConfigField(self.item_type, None)
                        return [temp_field._convert_value(item) for item in items]
                    return items
                elif isinstance(value, list):
                    return value
                else:
                    raise ValueError(f"Cannot convert {value} to list")
            elif self.field_type == FieldType.PATH:
                return str(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert value {value} to {self.field_type.value}: {str(e)}"
            )
