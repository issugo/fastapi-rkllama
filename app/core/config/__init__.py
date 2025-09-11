import logging
from typing import Dict, Any

from core.config.ConfigSectionSchema import ConfigSectionSchema

logger = logging.getLogger("rkllama.config")

class ConfigException(Exception):
    pass

class ConfigFieldException(ConfigException):
    pass

class IncrementalConfigSchema(object):
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


