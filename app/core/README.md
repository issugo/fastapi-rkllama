# Package `app/core`

## Rôle
Contient toute la logique métier centrale de l'application : conversion de formats API, backends d'inférence, gestion des modèles, configuration et orchestration des workers.

## Structure

```
core/
├── api/                # Conversion entre formats d'API
├── backends/           # Backends d'inférence matérielle (RKLLM, RKNN)
├── config/             # Configuration de l'application
├── model/              # Gestion des modèles LLM
└── processing/         # Orchestration des workers et traitement
```

## Sous-packages

### `api/`
Gère la conversion bidirectionnelle entre les différents formats d'API (Ollama, OpenAI, RKLLama).

### `backends/`
Implémente l'interface avec les bibliothèques matérielles pour l'inférence sur Rockchip.

### `config/`
Système de configuration centralisé utilisant Pydantic pour la validation.

### `model/`
Gestion complète du cycle de vie des modèles : téléchargement, métadonnées, stockage, conversion.

### `processing/`
Architecture multi-processus pour l'orchestration des workers d'inférence et la gestion des requêtes.

## Dépendances principales
- `pydantic` : Validation et sérialisation des données
- `fastapi` : Framework web
- Bibliothèques Rockchip : `rkllm`, `rknn`

## Architecture
Le package `core` suit une architecture en couches :
1. **API Layer** (`core.api`) : Conversion de formats
2. **Model Layer** (`core.model`) : Gestion des modèles
3. **Processing Layer** (`core.processing`) : Orchestration
4. **Backend Layer** (`core.backends`) : Inférence matérielle
5. **Config Layer** (`core.config`) : Configuration
