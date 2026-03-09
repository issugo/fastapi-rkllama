# Package `app/core/api/parameters`

## Rôle
Définit tous les modèles Pydantic pour les requêtes, réponses et structures de données communes des différentes APIs.

## Fichiers

### Modèles communs

#### `commons.py`
**Classes** :
- `Message` : Message dans une conversation (rôle + contenu)
- `TextContent` : Contenu textuel
- `ImageContent` : Contenu image (URL ou base64)
- Autres structures communes entre APIs

### Modèles Ollama

#### `ollama_commons.py`
- Structures de données communes Ollama (Options, etc.)

#### `ollama_requests.py`
**Classes de requêtes** :
- `OllamaPullRequest` : Téléchargement de modèle
- `OllamaChatRequest` : Requête de chat
- `OllamaGenerateRequest` : Requête de génération
- `OllamaEmbeddingRequest` : Requête d'embedding
- `OllamaPushRequest` : Push de modèle
- `OllamaCreateRequest` : Création de modèle
- `OllamaCopyRequest` : Copie de modèle
- `OllamaDeleteRequest` : Suppression de modèle
- `OllamaShowRequest` : Affichage infos modèle

#### `ollama_responses.py`
**Classes de réponses** :
- `OllamaChatResponse` : Réponse de chat
- `OllamaGenerateResponse` : Réponse de génération
- `OllamaEmbeddingResponse` : Réponse d'embedding
- `OllamaListResponse` : Liste de modèles
- `OllamaShowResponse` : Infos de modèle

### Modèles OpenAI

#### `openai_commons.py`
- Structures communes OpenAI (choices, usage, etc.)

#### `openai_requests.py`
**Classes de requêtes** :
- `ChatCompletionRequest` : Requête de chat completion
- `CompletionRequest` : Requête de text completion
- `EmbeddingRequest` : Requête d'embedding
- `ModerationRequest` : Requête de modération
- `ImageGenerationRequest` : Génération d'image
- `ImageEditRequest` : Édition d'image
- `ImageVariationRequest` : Variation d'image
- `VisionRequest` : Analyse d'image

#### `openai_responses.py`
**Classes de réponses** :
- `ChatCompletionResponse` : Réponse de chat
- `CompletionResponse` : Réponse de complétion
- `EmbeddingResponse` : Réponse d'embedding
- `ModerationResponse` : Réponse de modération
- `OpenAIModel` : Modèle OpenAI

### Modèles RKLLama

#### `rkllama_commons.py`
- Structures communes RKLLama

#### `rkllama_requests.py`
**Classes de requêtes** :
- `RKPullRequest` : Requête de pull de modèle

#### `rkllama_responses.py`
**Classes de réponses** :
- Réponses spécifiques RKLLama

### Sous-package converter

#### `converter/ConversionConfig.py`
**Classe** : `ConversionConfig`
- Configuration pour la conversion de modèles HuggingFace → RKLLM

## Architecture
Tous les modèles utilisent Pydantic pour :
- Validation automatique des données
- Sérialisation/désérialisation JSON
- Documentation automatique via OpenAPI/Swagger
- Typage fort

## Dépendances
- `pydantic` : Validation et modèles de données
- `typing` : Annotations de types
