# Package `app/api`

## Rôle
Point d'entrée de l'API FastAPI. Enregistre et expose tous les routeurs d'API.

## Structure

```
api/
├── __init__.py           # Enregistrement des routers
└── routes/               # Implémentation des routes
```

## Fichiers principaux

### `__init__.py`
- **Rôle** : Configure le routeur principal de l'API
- **Composants** :
  - `api_router` : `APIRouter` - Routeur FastAPI principal qui agrège tous les sous-routeurs
  - `logger` : Logger pour le package API

### Routers enregistrés
1. **`rkllama.router`** : API native RKLLama
2. **`ollama_new.router`** : API compatible Ollama
3. **`openai_new.router`** : API compatible OpenAI
4. **`rkllm_converter.router`** : Endpoints de conversion de modèles

## Dépendances
- `fastapi` : Framework web
- `app.api.routes.*` : Modules de routes

## Utilisation
Le `api_router` est importé et inclus dans l'application principale via `app.include_router(api_router)` dans `main.py`.
