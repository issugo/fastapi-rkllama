# Package `app/core/model/storage_helpers`

## Rôle
Fournit les classes pour télécharger et gérer le stockage de modèles depuis différentes sources (HuggingFace, Ollama, locale).

## Fichiers principaux

### `PullSupplier.py`
**Classe** : `PullSupplier(ABC)`
- **Rôle** : Classe abstraite de base pour les fournisseurs de téléchargement
- **Méthodes abstraites** :
  - `logger() -> Logger` : Logger pour le fournisseur
  - `check_params()` : Valide les paramètres
  - `model_data() -> dict` : Retourne les données du modèle
  - `model_type() -> ModelType` : Retourne le type de modèle

### `OllamaPullSupplier.py`
**Classe** : `OllamaPullSupplier(PullSupplier)`
- **Rôle** : Télécharge des modèles depuis Ollama Registry
- **Attributs** :
  - `model_name: str` : Nom du modèle Ollama
  - `registry: str` : URL du registre
  - `insecure: bool` : Autoriser connexions non-sécurisées
  
- **Méthodes** :
  - `parse_model_name()` : Parse le nom du modèle
  - `fetch_manifest()` : Récupère le manifeste Ollama
  - `download_blob(digest)` : Télécharge un blob par digest
  - `pull()` : Télécharge le modèle complet

### `RKPullSupplier.py`
**Classe** : `RKPullSupplier(PullSupplier)`
- **Rôle** : Télécharge des modèles au format RKLLama
- **Méthodes** :
  - `pull_from_local()` : Copie depuis chemin local
  - `pull_from_huggingface()` : Télécharge depuis HuggingFace
  - `pull_from_url()` : Télécharge depuis URL

### `StorageHelper.py`
**Classe** : `StorageHelper(ABC)`
- **Rôle** : Classe abstraite pour helpers de stockage
- **Méthodes abstraites** :
  - `save_model(model_data)` : Sauvegarde un modèle
  - `load_model(model_id)` : Charge un modèle
  - `delete_model(model_id)` : Supprime un modèle
  - `list_models()` : Liste les modèles

### `OllamaStorageHelper.py`
**Classe** : `OllamaStorageHelper(StorageHelper)`
- **Rôle** : Gère le stockage au format Ollama
- **Méthodes** :
  - `save_manifest(manifest, model_name)` : Sauvegarde le manifeste
  - `save_blob(blob_data, digest)` : Sauvegarde un blob
  - `get_model_path(model_name)` : Retourne le chemin du modèle
  - `create_modelfile(model_info)` : Crée un fichier Modelfile

### `OllamaModelStorageHelper.py`
**Classe** : `OllamaModelStorageHelper`
- Helper spécialisé pour modèles Ollama

### `RkllamaStorageHelper.py`
**Classe** : `RkllamaStorageHelper(StorageHelper)`
- **Rôle** : Gère le stockage au format RKLLama
- **Méthodes** :
  - `save_rkllm_file(file_data, model_name)` : Sauvegarde un fichier RKLLM
  - `create_metadata(model_info)` : Crée les métadonnées

### `HuggingfaceFileSystem.py`
**Classe** : `HuggingfaceFileSystem`
- **Rôle** : Interface avec le filesystem HuggingFace
- **Méthodes** :
  - `download_file(repo_id, filename)` : Télécharge un fichier
  - `list_files(repo_id)` : Liste les fichiers d'un dépôt
  - `get_file_info(repo_id, filename)` : Infos sur un fichier

### `OllamaFileSystem.py`
**Classe** : `OllamaFileSystem`
- **Rôle** : Interface avec le filesystem Ollama
- **Méthodes** :
  - `pull_model(model_name)` : Pull un modèle Ollama
  - `get_blob(digest)` : Récupère un blob
  - `verify_digest(blob_data, expected_digest)` : Vérifie un digest

### `SupplierFileInfo.py`
**Enum** : `Supplier`
- `HUGGINGFACE` : HuggingFace Hub
- `OLLAMA` : Ollama Registry
- `LOCAL` : Système de fichiers local
- `URL` : URL directe

### `model_pull.py`
**Fonctions utilitaires** :
- `pull_model(model_reference, supplier_type)` : Fonction principale de pull
- `detect_supplier(model_reference)` : Détecte automatiquement le type de fournisseur
- `validate_model_reference(reference, supplier)` : Valide une référence de modèle

## Architecture
Le package suit un pattern Strategy/Factory :
1. **PullSupplier** : Interface commune pour téléchargement
2. **Suppliers spécifiques** : Implémentations pour chaque source
3. **StorageHelper** : Interface commune pour stockage
4. **Helpers spécifiques** : Implémentations pour chaque format
5. **FileSystem** : Abstractions pour accès aux fichiers

## Flux de téléchargement
1. Détection du type de fournisseur depuis la référence du modèle
2. Création du PullSupplier approprié
3. Validation des paramètres
4. Téléchargement du modèle
5. Sauvegarde via le StorageHelper approprié

## Dépendances
- `huggingface_hub` : Téléchargement depuis HuggingFace
- `requests` : Téléchargements HTTP
- `pathlib` : Gestion de chemins
- `hashlib` : Vérification de digests
