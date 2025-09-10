from typing import Dict, List, Any

from core.config.FieldType import FieldType
from core.config.ConfigField import ConfigField


class ConfigSectionSchema:
    """Schema definition for a configuration section"""

    def __init__(self, description: str = ""):
        self.description = description
        self.fields: Dict[str, ConfigField] = {}

    def add_field(self, name: str, field: ConfigField):
        """Add a field to this section schema"""
        self.fields[name] = field
        return self

    def string(
        self,
        name: str,
        default: str = "",
        description: str = "",
        options: List[str] = None,
        required: bool = False,
    ):
        """Add a string field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.STRING, default, description, options=options, required=required
        )
        return self

    def integer(
        self,
        name: str,
        default: int = 0,
        description: str = "",
        min_value: int = None,
        max_value: int = None,
        required: bool = False,
    ):
        """Add an integer field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.INTEGER,
            default,
            description,
            min_value=min_value,
            max_value=max_value,
            required=required,
        )
        return self

    def float(
        self,
        name: str,
        default: float = 0.0,
        description: str = "",
        min_value: float = None,
        max_value: float = None,
        required: bool = False,
    ):
        """Add a float field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.FLOAT,
            default,
            description,
            min_value=min_value,
            max_value=max_value,
            required=required,
        )
        return self

    def boolean(
        self,
        name: str,
        default: bool = False,
        description: str = "",
        required: bool = False,
    ):
        """Add a boolean field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.BOOLEAN, default, description, required=required
        )
        return self

    def list(
        self,
        name: str,
        default: List = None,
        description: str = "",
        item_type: FieldType = None,
        required: bool = False,
    ):
        """Add a list field to this section schema"""
        if default is None:
            default = []
        self.fields[name] = ConfigField(
            FieldType.LIST, default, description, item_type=item_type, required=required
        )
        return self

    def path(
        self,
        name: str,
        default: str = "",
        description: str = "",
        required: bool = False,
    ):
        """Add a path field to this section schema"""
        self.fields[name] = ConfigField(
            FieldType.PATH, default, description, required=required
        )
        return self

    def validate_section(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate values against this section schema"""
        validated = {}

        # First, apply defaults for all fields
        for name, field in self.fields.items():
            validated[name] = field.default

        # Then override with provided values
        if values:
            for name, value in values.items():
                if name in self.fields:
                    try:
                        validated[name] = self.fields[name].validate(value)
                    except ValueError as e:
                        raise ValueError(f"Validation error for {name}: {str(e)}")
                else:
                    # Keep unknown fields, but don't validate them
                    validated[name] = value

        return validated
