# Package `app/api/routes`

## Rôle
Implémente les routes et endpoints de l'API pour les différents formats supportés (RKLLama, Ollama, OpenAI).

## Fichiers

### `rkllama.py` - API Native RKLLama
**Fonctions principales** :
- `pull_model()` : `POST /pull` - Télécharge un modèle
- `list_models()` : `GET /models` - Liste tous les modèles disponibles
- `get_model()` : `GET /model/{model_id}` - Récupère les infos d'un modèle
- `load_model_route()` : `POST /load_model` - Charge un modèle en mémoire
- `unload_model_route()` : `POST /unload_model` - Décharge un modèle de la mémoire
- `recevoir_message()` : `POST /generate` - Génère du texte
- `Rm_model()` : `DELETE /rm` - Supprime un modèle
- `get_current_models()` : `GET /current_models` - Liste les modèles chargés
- `create_model()` : `POST /create` - Crée un modèle depuis un Modelfile

**Classes internes** :
- `LocalRKPullSupplier` : Implémente le téléchargement de modèles depuis différentes sources

### `ollama_new.py` - API Compatible Ollama
**Fonctions principales** :
- `pull_model()` : `POST /api/pull` - Télécharge un modèle (compatible Ollama)
- `list_models()` : `GET /api/tags` - Liste les modèles
- `show_model()` : `POST /api/show` - Affiche les infos détaillées d'un modèle
- `show_model_by_id()` : `GET /api/show/{model_id}` - Affiche un modèle par ID
- `generate()` : `POST /api/generate` - Génère une complétion
- `chat()` : `POST /api/chat` - Génère une réponse de chat
- `embeddings()` : `POST /api/embeddings` - Génère des embeddings
- `push_model()` : `POST /api/push` - Pousse un modèle vers un registre
- `create_model()` : `POST /api/create` - Crée un modèle
- `copy_model()` : `POST /api/copy` - Copie un modèle
- `delete_model()` : `DELETE /api/delete` - Supprime un modèle

**Classes internes** :
- `LocalOllamaPullSupplier` : Gère le téléchargement de modèles au format Ollama

### `openai_new.py` - API Compatible OpenAI
**Fonctions principales** :
- `list_models()` : `GET /v1/models` - Liste les modèles disponibles
- `get_model()` : `GET /v1/models/{model_id}` - Récupère un modèle par ID
- `create_chat_completion()` : `POST /v1/chat/completions` - Crée une complétion de chat
- `create_completion()` : `POST /v1/completions` - Crée une complétion de texte
- `create_embeddings()` : `POST /v1/embeddings` - Génère des embeddings
- `create_moderation()` : `POST /v1/moderations` - Modération de contenu
- `create_image()` : `POST /v1/images/generations` - Génère une image
- `edit_image()` : `POST /v1/images/edits` - Édite une image
- `create_image_variation()` : `POST /v1/images/variations` - Crée une variation d'image
- `create_transcription()` : `POST /v1/audio/transcriptions` - Transcription audio
- `create_translation()` : `POST /v1/audio/translations` - Traduction audio
- `vision_completion()` : `POST /v1/vision/completions` - Analyse d'images

**Fonctions helper** :
- `stream_chat_response()` : Stream les réponses de chat en format SSE
- `stream_completion_response()` : Stream les complétions en format SSE

### Fichiers legacy
- `ollama.py` : Ancienne version de l'API Ollama (deprecated)
- `rkllama_old.py` : Ancienne version de l'API RKLLama (deprecated)

## Dépendances
- `fastapi` : Framework de routes
- `core.api.conversion.*` : Convertisseurs de formats
- `core.api.parameters.*` : Modèles de requêtes/réponses Pydantic
- `core.model.*` : Gestion des modèles
- `core.processing.WorkerManager` : Gestion des workers d'inférence

## Architecture
Chaque fichier de route :
1. Reçoit une requête au format spécifique (RKLLama/Ollama/OpenAI)
2. Convertit les paramètres vers le format interne
3. Appelle le `WorkerManager` pour l'inférence
4. Convertit la réponse vers le format attendu
5. Retourne la réponse (streaming ou non)
