import argparse
import logging
from typing import TypeVar, Any, Optional, Union

from core.config.RKLLAMAConfig import RKLLAMAConfig
from core.config.FieldType import FieldType

logger = logging.getLogger("rkllama.config")
T = TypeVar("T")
config = RKLLAMAConfig()


def get(
    section: str,
    key: str,
    default: Any = None,
    as_type: Optional[Union[FieldType, type]] = None,
) -> Any:
    """
    Retrieves a configuration value with optional type conversion.

    Examples:
        # Get a string value
        name = get("app", "name", "DefaultApp")

        # Get with type conversion
        port = get("server", "port", 8080, as_type=int)
        debug = get("server", "debug", False, as_type=bool)
        hosts = get("server", "allowed_hosts", [], as_type=list)
    """
    return config.get(section, key, default, as_type)


def set(section: str, key: str, value: Any):
    """Set a configuration value"""
    config.set(section, key, value)


def get_path(key: str, default: Any = None) -> str:
    """Get a path configuration value"""
    return config.get_path(key, default)


def display():
    """Display the current configuration"""
    config.display()


def validate():
    """Validate the current configuration"""
    return config.validate()


def load_args(args: argparse.Namespace):
    """Load configuration from command-line arguments"""
    config.load_args(args)


def save_to_project_ini():
    """Save current configuration to project INI file"""
    config.save_to_project_ini()


def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return config.get("server", "debug", False, as_type=bool)


def reload_config():
    """Reload configuration from all sources"""
    config.reload_config()
