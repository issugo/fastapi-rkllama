# Package `app/core/api/parameters`

## Rôle
Définit tous les modèles Pydantic pour les requêtes, réponses et structures de données communes des différentes APIs.

le contenu du package `app.core.api.parameters` respecte les règles suivantes :
- Les classes sont définies dans des fichiers séparés pour chaque API (ollama.py, openai.py, rkllama.py)
- Les classes sont nommées de manière explicite et cohérente avec le contexte de l'API
- Les classes sont documentées avec des commentaires explicites sur leurs attributs et leurs utilisations
- Les classes sont héritées de Pydantic.BaseModel pour garantir la validation des données
- Les classes sont utilisées dans les routes correspondantes pour la validation des requêtes et la construction des réponses
- les types communs entre les request et les response pour l'API Ollama sont définis dans `ollama_commons.py`
- les types communs entre les request et les response pour l'API OpenAI sont définis dans `openai_commons.py`
- les types communs entre les request et les response pour l'API Rkllama sont définis dans `rkllama_commons.py`
- les types communs entre l'API Ollama et l'API OpenAI sont définis dans `commons.py`
- les types communs entre l'API Rkllama et les APIs Ollama et OpenAI sont définis dans `commons.py`
- Les types de données de l'API Ollama doivent être cohérents avec les spécifications de l'API Ollama (disponible sur internet).
- Les types de données de l'API OpenAI doivent être cohérents avec les spécifications de l'API OpenAI (disponible sur internet).

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
