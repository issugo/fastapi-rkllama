import argparse
import datetime
import os
from pathlib import Path
from typing import Tuple, Optional, Any, Union, List, Annotated, get_type_hints
from core.config.warnings import deprecated

import yaml
from pydantic import BaseModel, Field, json
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

from core.config import logger
from core.config.ServerConfig import ServerConfig, Server
from core.config.PathsConfig import PathsConfig, Paths, PATH_KEY
from core.config.DefaultModelConfig import DefaultModelConfig, DefaultConfig
from core.config.PlatformConfig import PlatformConfig, PlatformProcessor, Platform

def system_config_paths():
    return [
        Path("/etc/rkllama/rkllama.yml"),
        Path("/etc/rkllama.yml"),
        Path("/usr/local/etc/rkllama.yml"),
        Path.cwd() / "system" / "rkllama.yml",
    ]

def user_config_paths():
    return [
        Path.home() / ".config" / "rkllama" / "rkllama.yml",
        Path.home() / ".config" / "rkllama.yml",
        Path.home() / ".rkllama.yml",
    ]

def project_config_paths():
    return [
        Path.cwd() / "rkllama.yml",
        Path.cwd() / "config" / "rkllama.yml",
    ]


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """
    A simple settings source class that loads variables from a JSON file
    at the project's root.

    Here we happen to choose to use the `env_file_encoding` from Config
    when reading `config.json`
    """

    path_content_map: dict[Path, dict] = {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        file_content_yaml = {}
        encoding = self.config.get('env_file_encoding')
        for paths_getter in [system_config_paths, user_config_paths, project_config_paths]:
            file_paths = paths_getter()
            for file_path in file_paths:
                if file_path not in self.path_content_map:
                    if file_path.exists():
                        with open(file_path, encoding=encoding) as f:
                            file_content_yaml = yaml.load(f, Loader=yaml.SafeLoader)
                            self.path_content_map[file_path] = file_content_yaml
                else:
                    file_content_yaml = self.path_content_map[file_path]
        field_value = file_content_yaml.get(field_name)
        return field_value, field_name, False

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        d: dict[str, Any] = {}

        for field_name, field in self.settings_cls.model_fields.items():
            field_value, field_key, value_is_complex = self.get_field_value(
                field, field_name
            )
            field_value = self.prepare_field_value(
                field_name, field, field_value, value_is_complex
            )
            if field_value is not None:
                d[field_key] = field_value

        return d



class RKLLAMASettings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, env_prefix='RKLLAMA_', env_nested_delimiter='__', env_file=('.env', '.env.prod'), env_file_encoding='utf-8')

    app_root: Path = Field(default=Path(os.getcwd()), description="Application root directory")
    server: Server = Field(default=Server(), description="Server configuration settings")
    paths: Paths = Field(default=Paths(), description="Paths configuration settings")
    model: DefaultConfig = Field(default=DefaultConfig(), description="DefaultConfig configuration settings")
    platform: Platform = Field(default=Platform(), description="Platform configuration settings")

    _path_cache: dict = {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @property
    def settings(self) -> dict:
        return self.model_dump()

    def resolve_path(self, path: str) -> str | None:
        """Resolve a path relative to the application root"""
        if not path:
            return None

        # Check if we have this path in the cache
        if path in self._path_cache:
            return self._path_cache[path]

        path_obj = Path(path)

        if path_obj.is_absolute():
            resolved = str(path_obj)
        elif "$" in path or "~" in path:
            # Check if path contains environment variables and expand them
            expanded_path = os.path.expanduser(os.path.expandvars(path))
            if os.path.isabs(expanded_path):
                resolved = expanded_path
            else:
                # Relative to app root after expansion
                resolved = str(self.app_root / expanded_path)
        else:
            # Relative to app root
            resolved = str(self.app_root / path)

        # Cache the result
        self._path_cache[path] = resolved
        return resolved

    def get_path(self, key: str | PATH_KEY, default: Any = None) -> str:
        """
        Retrieves a path configuration and resolves it.
        Path resolution includes app_root and environment variable expansion.
        """
        if isinstance(key, PATH_KEY):
            key = key.value
        path = self.paths.__getattribute__(key)
        if path is None:
            path = default
        return self.resolve_path(path) if path else None

    def display(self):
        """Logs the current configuration values"""
        logger.info("Current RKLLAMA Configuration:")
        logger.info("\n" + yaml.dump(self.settings))

    def is_debug_mode(self) -> bool:
        """Checks if debug mode is enabled"""
        return self.server.debug


@deprecated("use core.config.RKLLAMASettings instead.", category=DeprecationWarning, stacklevel=2)
class RKLLAMAConfig(BaseModel):
    """Centralized configuration system for RKLLAMA"""
    _app_root: Path = Path(os.getcwd())
    _config_dir: Path = None
    _path_cache: dict = {}
    _type_cache: dict = {}
    _args: argparse.Namespace = None

    server: Annotated[ServerConfig, Field(description="Server configuration settings")] = ServerConfig()
    paths: Annotated[PathsConfig, Field(description="Path configuration")] = PathsConfig()
    model: Annotated[DefaultModelConfig, Field(description="Model configuration")] = DefaultModelConfig()
    platform: Annotated[PlatformConfig, Field(description="Platform configuration")] = PlatformConfig()

    @deprecated("use core.config.RKLLAMASettings.settings instead.", category=DeprecationWarning, stacklevel=2)
    @property
    def config(self) -> dict:
        return self.model_dump()

    def __init__(self, **data: Any):
        super().__init__(**data)

        app_root: Path = data.get("app_root")
        args: argparse.Namespace = data.get("args")

        if app_root is None:
            app_root = Path(os.getcwd())
        self._app_root = app_root
        logger.debug(f"app_root={self._app_root}")
        self._config_dir = self._app_root / "config"

        # store args
        self._args = args
        logger.debug(f"args={self._args}")

        # Create the config directory if it doesn't exist
        os.makedirs(self._config_dir, exist_ok=True)

        self.reload_config()

    def _update_dict(self, u: dict, model: BaseModel = None):
        if model is None:
            return self._update_dict(model=self, u=u)
        else:
            for k, v in u.items():
                if isinstance(v, dict):
                    model.__setattr__(k, self._update_dict(model=model.__getattr__(k), u=v))
                else:
                    if k in model.__dict__:
                        model.__setattr__(k, v)
            return model

    def _get_field_info(
        self, section: str, key: str
    ) -> Any:
        """
        Get field type
        """
        type_hints: dict = get_type_hints(self.__getattribute__(section).__class__)
        return type_hints.get(key)

    def _infer_and_convert_type(self, section: str, key: str, value: str) -> Any:
        """
        Converts string values to appropriate Python types.

        Uses schema if available, otherwise applies heuristic type detection
        for booleans, numbers, and lists.
        """
        # Handle None values
        if value is None:
            return None

        # Check if we already know the type from schema
        field_type = self._get_field_info(section, key)
        if field_type is not None:
            match field_type:
                case bool():
                    return value.lower() in ["1", "true", "yes", "on"]
                case int():
                    return int(value)
                case float():
                    return float(value)
                case list():
                    return value.split(",")
                case PlatformProcessor():
                    return PlatformProcessor(value)
                case _:
                    return value
        return None

    def _write_defaults(self):
        """Write default values from schema and creates default.yml file"""
        default_config = self.model_dump()

        # Write default configuration to file if it doesn't exist
        default_yml_path = self._config_dir / "default.yml"
        if not default_yml_path.exists():
            with open(default_yml_path, "w") as f:
                yaml.dump(default_config, f)

    def _load_config_file(self, config_path: Union[str, Path]):
        """
        Loads and parses an INI configuration file.
        Performs type conversion during loading.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)

        if not config_path.exists():
            return None

        logger.debug(f"Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        self._update_dict(config)

    def _load_system_ini(self):
        """Load system-wide configuration"""
        system_config_paths = [
            Path("/etc/rkllama/rkllama.yml"),
            Path("/etc/rkllama.yml"),
            Path("/usr/local/etc/rkllama.yml"),
            self._app_root / "system" / "rkllama.yml",
        ]

        for path in system_config_paths:
            if path.exists():
                self._load_config_file(path)
                logger.debug(f"Loaded system configuration from: {path}")

    def _load_user_ini(self):
        """Load user-specific configuration"""
        user_config_paths = [
            Path.home() / ".config" / "rkllama" / "rkllama.yml",
            Path.home() / ".config" / "rkllama.yml",
            Path.home() / ".rkllama.yml",
        ]

        for path in user_config_paths:
            if path.exists():
                self._load_config_file(path)
                logger.debug(f"Loaded user configuration from: {path}")

    def _load_project_ini(self):
        """Load project-specific configuration"""
        project_config_paths = [
            self._app_root / "rkllama.yml",
            self._app_root / "config" / "rkllama.yml",
        ]

        for path in project_config_paths:
            if path.exists():
                self._load_config_file(path)
                logger.debug(f"Loaded project configuration from: {path}")

    def _load_env_vars(self):
        """
        Load configuration from environment variables.
        Environment variables override ini files.
        """
        sections = self.config.keys()
        # Pattern: RKLLAMA_SECTION_KEY
        for env_var, value in os.environ.items():
            if not env_var.startswith("RKLLAMA_"):
                continue

            # Special case for RKLLAMA_DEBUG environment variable
            if env_var == "RKLLAMA_DEBUG":
                if value.lower() in ["1", "true", "yes", "on"]:
                    self.server.debug = True
                elif value.lower() in ["0", "false", "no", "off"]:
                    self.server.debug = False
                continue

            parts = env_var.split("_")
            if len(parts) < 3:
                continue

            section = parts[1].lower()
            key = "_".join(parts[2:]).lower()

            if section not in sections:
                continue

            # Convert environment variable value to appropriate type
            typed_value = self._infer_and_convert_type(section, key, value)
            if typed_value is None:
                logger.warning(f"Invalid value for {env_var}: {value}")

            # Environment variables take precedence over ini files
            self.__getattribute__(section).__setattr__(key, typed_value)
            logger.debug(f"Loaded config from environment: {env_var}={typed_value}")

    @deprecated("use core.config.RKLLAMASettings as pydantic settings instead.", category=DeprecationWarning, stacklevel=2)
    def load_args(self, args: argparse.Namespace):
        """
        Load configuration from command-line arguments.
        Command-line args have the highest priority.
        """

        # Extract all args and apply them
        if args:
            # Handle common explicit arguments
            if hasattr(args, "port") and args.port is not None:
                self.server.port = int(args.port)

            if hasattr(args, "debug") and args.debug:
                self.server.debug = True

            if hasattr(args, "processor") and args.processor:
                self.platform.processor = PlatformProcessor(args.processor)

            if hasattr(args, "config") and args.config:
                # Load a custom config file with the highest priority
                custom_config = Path(args.config)
                if custom_config.exists():
                    self._load_config_file(custom_config)
                else:
                    logger.warning(f"Specified config file not found: {args.config}")

    @deprecated("use core.config.RKLLAMASettings.resolve_path instead.", category=DeprecationWarning, stacklevel=2)
    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to the application root"""
        if not path:
            return None

        # Check if we have this path in the cache
        if path in self._path_cache:
            return self._path_cache[path]

        path_obj = Path(path)

        if path_obj.is_absolute():
            resolved = str(path_obj)
        elif "$" in path or "~" in path:
            # Check if path contains environment variables and expand them
            expanded_path = os.path.expanduser(os.path.expandvars(path))
            if os.path.isabs(expanded_path):
                resolved = expanded_path
            else:
                # Relative to app root after expansion
                resolved = str(self._app_root / expanded_path)
        else:
            # Relative to app root
            resolved = str(self._app_root / path)

        # Cache the result
        self._path_cache[path] = resolved
        return resolved

    def _clear_path_cache(self):
        """Clear the path resolution cache"""
        self._path_cache = {}


    @deprecated("use core.config.RKLLAMASettings.get_path instead.", category=DeprecationWarning, stacklevel=2)
    def get_path(self, key: str, default: Any = None) -> str:
        """
        Retrieves a path configuration and resolves it.
        Path resolution includes app_root and environment variable expansion.
        """
        path = self.paths.__getattribute__(key)
        if path is None:
            path = default
        return self.resolve_path(path) if path else None

    def _generate_shell_config(self):
        """
        Creates a shell script with environment variables.
        Useful for sourcing in shell scripts or CI/CD pipelines.
        """
        config_env_path = self._config_dir / "config.env"

        lines = [
            "#!/bin/sh",
            "# Auto-generated shell configuration for RKLLAMA",
            f"# Generated at: {datetime.datetime.now().isoformat()}",
            "",
            "# Application root",
            f'RKLLAMA_ROOT="{self._app_root}"',
            "",
        ]

        # Add all configuration values
        for section, values in self.config.items():
            lines.append(f"# {section.upper()} configuration")
            for key, value in (values if isinstance(values, dict) else values.__dict__).items():
                # Convert to shell variable format
                env_var = f"RKLLAMA_{section.upper()}_{key.upper()}"
                # Convert typed values to string representation for shell
                str_value = str(value)

                # Handle special cases for shell variables
                if isinstance(value, bool):
                    str_value = "1" if value else "0"
                elif isinstance(value, list):
                    str_value = ",".join(str(item) for item in value)

                lines.append(f'{env_var}="{str_value}"')

                # Special case for paths - add resolved paths as well
                if section == "paths":
                    resolved_path = self.resolve_path(str_value)
                    lines.append(f'{env_var}_RESOLVED="{resolved_path}"')

            lines.append("")

        # Write to file
        with open(config_env_path, "w") as f:
            f.write("\n".join(lines))

        # Make the file executable
        os.chmod(config_env_path, 0o755)

        logger.debug(f"Generated shell configuration: {config_env_path}")

    @deprecated("use core.config.RKLLAMASettings.display instead.", category=DeprecationWarning, stacklevel=2)
    def display(self):
        """Logs the current configuration values"""
        logger.info("Current RKLLAMA Configuration:")
        logger.info(yaml.dump(self.config))

    @deprecated("use core.config.RKLLAMASettings.is_debug_mode instead.", category=DeprecationWarning, stacklevel=2)
    def is_debug_mode(self) -> bool:
        """Checks if debug mode is enabled"""
        return self.server.debug

    @deprecated("reload_config is inefficient using docker.", category=DeprecationWarning, stacklevel=2)
    def reload_config(self):
        """
        Reloads all configuration from all sources.
        Maintains priority order and preserves command-line arguments.
        """

        # Clear current config and caches
        self._clear_path_cache()
        self._type_cache = {}

        # Reload in proper priority order
        self._write_defaults()
        self._load_system_ini()
        self._load_user_ini()
        self._load_project_ini()
        self._load_env_vars()
        self.load_args(self._args)

        if self.is_debug_mode():
            self.display()

        # Re-generate shell config
        self._generate_shell_config()

        logger.debug("Configuration reloaded")


