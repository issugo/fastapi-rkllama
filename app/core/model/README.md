# Package `app/core/model`

## Rôle
Gestion complète du cycle de vie des modèles LLM : téléchargement, métadonnées, stockage, configuration et conversion.

## Fichiers principaux

### `Model.py`
**Classes** :

- `ModelSharedData(BaseModel)` : Données partagées entre processus pour un modèle
  - **Attributs** :
    - `global_status: int` : Statut de génération (-1=idle, 0=running, 1=finished, <0=error)
    - `global_text: List[str]` : Tokens générés

- `Model(BaseModel)` : Classe principale représentant un modèle
  - **Attributs** :
    - `id: str` : Identifiant unique
    - `st_atime, st_mtime, st_ctime: float` : Timestamps fichier
    - `size: int` : Taille du modèle
    - `digest: str` : Hash du modèle
    - `model_path: ModelPath` : Chemin du modèle
    - `model_info: ModelInfo` : Métadonnées du modèle
    - `shared_data: ModelSharedData` : Données partagées
  
  - **Méthodes** :
    - `from_model_path(cls, model_path) -> Model` : Crée un modèle depuis un chemin
    - `load(cls, model_path) -> Model` : Charge un modèle avec ses métadonnées
    - `save()` : Sauvegarde les métadonnées du modèle
    - `list(cls) -> List[Model]` : Liste tous les modèles disponibles
    - `unload()` : Décharge le modèle de la mémoire
    - `clean_metadata(cls, model_path)` : Nettoie les métadonnées
    - `clean(cls, model_path)` : Nettoie complètement un modèle

### `ModelPath.py`
**Classes** :
- `ModelPath(BaseModel)` : Représente un chemin de modèle
- `ModelDirError`, `ModelDirException`, `ModelNotFoundException`, `ModelException` : Exceptions

### `ModelInfo.py`
**Classe** : `ModelInfo(BaseModel)`
- Métadonnées complètes d'un modèle (nom, version, licence, etc.)

### `ModelMetadata.py`
**Classe** : `ModelMetadata(BaseModel)`
- Métadonnées étendues incluant configuration et historique

### `ModelConfig.py`
**Classe** : `ModelConfig(BaseModel)`
- Configuration d'exécution d'un modèle (paramètres d'inférence)

### `ModelFile.py`
**Classe** : `ModelFile(BaseModel)`
- Représente un fichier de modèle avec ses propriétés

### `ModelName.py`
**Classe** : `ModelName(BaseModel)`
- Parse et valide les noms de modèles (format registry/namespace/name:tag)

### `ModelType.py`
**Classes** :
- `FILE_TYPE(str, Enum)` : Types de fichiers (RKLLM, RKNN, Safetensors, etc.)
- `ModelType(str, Enum)` : Types de modèles
- `model_type(file_suffix) -> ModelType` : Détermine le type depuis l'extension

### `OllamaManifest.py`
**Classe** : `OllamaManifest(BaseModel)`
- Manifeste au format Ollama

### `HfFileInfo.py`
**Classe** : `HfFileInfo(BaseModel)`
- Informations sur un fichier HuggingFace

### `ModelLicense.py`
**Classe** : `ModelLicense(BaseModel)`
- Licence du modèle

### `SupplierModelInfo.py`
**Classe** : `SupplierModelInfo(BaseModel)`
- Informations fournisseur de modèle

### Fichiers de constantes
- `models_constants.py` : Constantes globales (types de modèles, templates, etc.)
- `special_tokens.py` : Tokens spéciaux pour différents modèles
- `suppliers_model_info.py` : Informations sur les fournisseurs de modèles

## Sous-packages

### `converter/`
Conversion de modèles entre formats (HuggingFace → RKLLM).

### `storage_helpers/`
Helpers pour téléchargement et stockage de modèles depuis différentes sources.

## Architecture
Le package suit une architecture en couches :
1. **Model** : Point d'entrée principal, orchestre les autres composants
2. **ModelPath** : Gestion des chemins et localisation
3. **ModelInfo/Metadata** : Métadonnées et configuration
4. **ModelFile** : Représentation des fichiers physiques
5. **Storage Helpers** : Téléchargement et gestion du stockage
6. **Converters** : Conversion entre formats

## Dépendances
- `pydantic` : Modèles de données
- `pathlib` : Gestion de chemins
- `huggingface_hub` : Interaction avec HuggingFace
- `hashlib` : Calcul de hash  
- Bibliothèques de conversion RKLLM
