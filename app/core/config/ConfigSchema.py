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


def create_rkllama_schema() -> ConfigSchema:
    """Create and return the RKLLAMA configuration schema"""
    schema = ConfigSchema()

    # Server section
    server = schema.add_section("server", description="Server configuration settings")
    server.integer("port", 8080, "Server port number", min_value=1, max_value=65535)
    server.string("host", "0.0.0.0", "Server host address")
    server.boolean("debug", False, "Enable debug mode")

    # Paths section
    paths = schema.add_section("paths", description="Path configuration")
    paths.path("models", "models", "Path to model files")
    paths.path("logs", "logs", "Path to log files")
    paths.path("data", "data", "Path to data files")
    paths.path("src", "src", "Path to source files")
    paths.path("lib", "lib", "Path to library files")
    paths.path("temp", "temp", "Path to temporary files")

    # Model section
    model = schema.add_section("model", description="Model configuration")
    model.string("default", "", "Default model to use")
    model.string("default_temperature", 0.5, "Default temperature for the model to use")
    model.string(
        "default_enable_thinking", False, "Default Enable Thinking for the model to use"
    )
    model.string(
        "default_num_ctx", 16384, "Default Context Length for the model to use"
    )
    model.string(
        "default_max_new_tokens", 16384, "Default Max New Tokens for the model to use"
    )
    model.string("default_top_k", 7, "Default Top K for the model to use")
    model.string("default_top_p", 0.5, "Default Top P for the model to use")
    model.string(
        "default_repeat_penalty", 1.1, "Default Repeat Penalty for the model to use"
    )
    model.string(
        "default_frequency_penalty",
        0.0,
        "Default Frequency Penalty for the model to use",
    )
    model.string(
        "default_presence_penalty", 0.0, "Default Presence Penalty for the model to use"
    )
    model.string("default_mirostat", 0, "Default Mirostat for the model to use")
    model.string("default_mirostat_tau", 3, "Default Mirostat Tau for the model to use")
    model.string(
        "default_mirostat_eta", 0.1, "Default Mirostat Eta for the model to use"
    )

    # Platform section
    platform = schema.add_section("platform", description="Platform configuration")
    platform.string(
        "processor", "rk3588", "Target processor", options=["rk3588", "rk3576"]
    )

    return schema
