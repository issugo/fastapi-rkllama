from typing import Dict, Any

from core.config.ConfigSectionSchema import ConfigSectionSchema


class ConfigSchema:
    """Schema definition for the entire configuration"""

    def __init__(self):
        self.sections: Dict[str, ConfigSectionSchema] = {}

    def add_section(
        self, name: str, section: ConfigSectionSchema = None, description: str = ""
    ):
        """Add a section to this schema"""
        if section is None:
            section = ConfigSectionSchema(description)
        self.sections[name] = section
        return section

    def get_section(self, name: str) -> ConfigSectionSchema:
        """Get a section from this schema"""
        return self.sections.get(name)

    def validate(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate a configuration against this schema"""
        validated = {}

        # First, apply defaults for all sections
        for section_name, section_schema in self.sections.items():
            validated[section_name] = section_schema.validate_section({})

        # Then override with provided values
        if config:
            for section_name, section_values in config.items():
                if section_name in self.sections:
                    validated[section_name] = self.sections[
                        section_name
                    ].validate_section(section_values)
                else:
                    # Keep unknown sections, but don't validate them
                    validated[section_name] = section_values

        return validated
