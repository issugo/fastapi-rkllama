from typing import Dict, Any

from core.config import ServerConfig, IncrementalConfigSchema
from core.config.ConfigSectionSchema import ConfigSectionSchema
from core.config.ModelConfig import ModelConfig
from core.config.PathsConfig import PathsConfig
from core.config.PlatformConfig import PlatformConfig
from core.config.ServerConfig import ServerConfig


class ConfigSchema(IncrementalConfigSchema):
    """Schema definition for the entire configuration"""


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


def create_rkllama_schema() -> ConfigSchema:
    """Create and return the RKLLAMA configuration schema"""
    schema = ConfigSchema()

    # Server section
    ServerConfig.add_schema(schema)

    # Paths section
    PathsConfig.add_schema(schema)

    # Model section
    ModelConfig.add_schema(schema)

    # Platform section
    PlatformConfig.add_schema(schema)

    return schema


