# Package `app/core/config`

## Rôle
Système de configuration centralisé pour l'application RKLLAMA, supportant plusieurs sources de configuration (YAML, variables d'environnement, arguments CLI).

## Fichiers

### `RKLLAMAConfig.py`
**Classes principales** :

- `YamlConfigSettingsSource(PydanticBaseSettingsSource)` : Source de configuration YAML personnalisée
  - **Méthodes** :
    - `get_field_value(field, field_name)` : Récupère la valeur d'un champ depuis YAML
    - `prepare_field_value(field_name, field, value, value_is_complex)` : Prépare la valeur
    - `__call__()` : Charge et retourne la configuration YAML

- `RKLLAMASettings(BaseSettings)` : Configuration principale utilisant Pydantic Settings
  - **Attributs** :
    - `app_root: Path` : Racine de l'application
    - `paths: PathsConfig` : Configuration des chemins
    - `server: ServerConfig` : Configuration du serveur
    - `platform: PlatformConfig` : Configuration de la plateforme
    - `default_model: DefaultModelConfig` : Configuration du modèle par défaut
    - `debug: bool` : Mode debug
  
  - **Méthodes** :
    - `settings_customise_sources()` : Personnalise les sources de configuration (YAML + env)
    - `resolve_path(path)` : Résout un chemin relatif à app_root
    - `get_path(key, default)` : Récupère et résout un chemin configuré
    - `display()` : Affiche la configuration
    - `is_debug_mode()` : Vérifie si le mode debug est activé

- `RKLLAMAConfig` : Classe de configuration legacy (deprecated)
  - Système de configuration INI avec priorités multiples
  - Support arguments CLI, variables d'environnement, fichiers INI

**Fonctions** :
- `system_config_paths()` : Retourne les chemins de config système
- `user_config_paths()` : Retourne les chemins de config utilisateur
- `project_config_paths()` : Retourne les chemins de config projet

### `PathsConfig.py`
**Classes** :

- `PATH_KEY(str, Enum)` : Énumération des clés de chemins
  - `APP_ROOT`, `CONFIG`, `LIB`, `LOGS`, `MODELS`, `TEMPORARY`, `CACHE`

- `Paths(BaseModel)` : Chemins de l'application
  - **Attributs** : `app_root`, `config`, `lib`, `logs`, `models`, `temporary`, `cache`

- `PathsConfig(BaseModel)` : Configuration des chemins
  - **Attributs** : `paths: Paths`

### `ServerConfig.py`
**Classes** :

- `Server(BaseModel)` : Configuration serveur
  - **Attributs** :
    - `host: str` : Hôte du serveur (défaut: "0.0.0.0")
    - `port: int` : Port du serveur (défaut: 8080)

- `ServerConfig(BaseModel)` : Wrapper de configuration serveur
  - **Attributs** : `server: Server`

### `PlatformConfig.py`
**Classes** :

- `PlatformProcessor(str, Enum)` : Processeurs supportés
  - `RK3588`, `RK3576`

- `Platform(BaseModel)` : Configuration plateforme
  - **Attributs** :
    - `name: str` : Nom de la plateforme
    - `processor: PlatformProcessor` : Type de processeur

- `PlatformConfig(BaseModel)` : Wrapper de configuration plateforme
  - **Attributs** : `platform: Platform`

### `DefaultModelConfig.py`
**Classes** :

- `DefaultConfig(BaseModel)` : Configuration du modèle par défaut
  - **Attributs** :
    - `load_on_startup: bool` : Charger au démarrage
    - `model_path: Optional[str]` : Chemin du modèle
    - `modelfile: Optional[str]` : Nom du modelfile

- `DefaultModelConfig(BaseModel)` : Wrapper de configuration modèle
  - **Attributs** : `default_model: DefaultConfig`

### `config_utils.py`
**Fonctions** :
- `get_settings() -> RKLLAMASettings` : Retourne l'instance singleton de configuration
- `get_path(key: PATH_KEY) -> Path` : Raccourci pour obtenir un chemin configuré

### `warnings.py`
**Décorations** :
- `@deprecated` : Décorator pour marquer des fonctions comme dépréciées

## Architecture
Le système de configuration suit cette hiérarchie de priorité (du plus faible au plus fort) :
1. **Valeurs par défaut** : Définies dans les modèles Pydantic
2. **Fichiers YAML** : `config/config.yaml`
3. **Variables d'environnement** : Préfixe `RKLLAMA_`
4. **Arguments CLI** : Passés au démarrage

Toutes les configurations utilisent Pydantic pour :
- Validation automatique des types
- Conversion de types
- Génération de schéma
- Documentation auto-générée

## Dépendances
- `pydantic` : Modèles de données et validation
- `pydantic-settings` : Gestion de configuration
- `yaml` : Parsing YAML
- `argparse` : Arguments CLI
