# Package `app/core/backends`

## Rôle
Implémente les backends d'inférence matérielle pour les processeurs Rockchip (RKLLM et RKNN).

## Structure

```
backends/
├── backend.py          # Classes de base et types
├── GlobalState.py      # État global du backend
├── rkllm/              # Backend RKLLM
└── rknn/               # Backend RKNN
```

## Fichiers principaux

### `backend.py`
**Classes** :

- `BackendType(str, Enum)` : Énumération des types de backend
  - `RKLLM` : Backend pour modèles RKLLM
  - `RKNN` : Backend pour modèles RKNN
  - **Méthode** : `from_model_type(cls, model_type: ModelType) -> BackendType` - Convertit un type de modèle en type de backend

- `BackendException(Exception)` : Exception pour les erreurs de backend

- `Backend` : Classe de base abstraite pour les backends
  - **Attributs** : `backend_type: BackendType`
  - **Méthodes** : 
    - `run(self, param)` : Exécute l'inférence (à implémenter)

**Constantes** :
- `BACKEND_SUPPORTED_LIB_VERSION` : Versions supportées pour chaque backend

### `GlobalState.py`
**Classes** :

- `GlobalState(BaseModel)` : État global partagé du backend
  - **Attributs** :
    - `rkllm_model: Optional[Model]` : Modèle actuellement chargé
  - **Propriétés** :
    - `current_model -> Union[ModelPath|None]` : Chemin du modèle chargé
    - `loaded_model_hfpath -> Union[str|None]` : Chemin HuggingFace du modèle chargé

**Variables globales** :
- `GLOBAL_STATE` : Instance unique de `GlobalState`

**Fonctions** :
- `unload_model()` : Décharge le modèle actuellement en mémoire

## Sous-packages

### `rkllm/`
Implémente le backend pour les modèles RKLLM (format natif Rockchip pour LLM).

### `rknn/`
Implémente le backend pour les modèles RKNN (format neural network Rockchip).

## Architecture
Le package utilise un pattern Strategy :
1. La classe abstraite `Backend` définit l'interface commune
2. Les sous-packages `rkllm/` et `rknn/` implémentent des backends spécifiques
3. `BackendType` permet de sélectionner le bon backend selon le type de modèle
4. `GlobalState` maintient l'état partagé entre les backends

## Dépendances
- `pydantic` : Modèles de données
- `enum` : Énumérations
- `core.model.*` : Types et gestion de modèles
- Bibliothèques Rockchip natives (RKLLM, RKNN)
