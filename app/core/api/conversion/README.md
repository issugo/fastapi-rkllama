# Package `app/core/api/conversion`

## Rôle
Implémente les convertisseurs bidirectionnels entre les formats d'API Ollama et OpenAI.

## Fichiers

### `ollama.py`
**Classe** : `OllamaConversions`
- **Rôle** : Convertit les requêtes Ollama vers le format OpenAI

**Méthodes principales** :
- `convert_chat_request(request: OllamaChatRequest) -> ChatCompletionRequest`
  - Convertit une requête de chat Ollama vers OpenAI
  
- `convert_generate_request(request: OllamaGenerateRequest) -> CompletionRequest`
  - Convertit une requête de génération Ollama vers OpenAI
  
- `convert_embedding_request(request: OllamaEmbeddingRequest) -> EmbeddingRequest`
  - Convertit une requête d'embedding Ollama vers OpenAI
  
- `convert_generate_with_images_to_vision(request: OllamaGenerateRequest, images: List[str]) -> VisionRequest`
  - Convertit une génération Ollama avec images vers requête vision OpenAI
  
- `convert_chat_with_images_to_vision(request: OllamaChatRequest, images: List[str]) -> VisionRequest`
  - Convertit un chat Ollama avec images vers requête vision OpenAI

### `openai.py`
**Classe** : `OpenAIConversions`
- **Rôle** : Convertit les requêtes OpenAI vers le format Ollama

**Méthodes principales** :
- `convert_chat_completion_request(request: ChatCompletionRequest) -> OllamaChatRequest`
  - Convertit une requête de chat OpenAI vers Ollama
  
- `convert_completion_request(request: CompletionRequest) -> OllamaGenerateRequest`
  - Convertit une requête de complétion OpenAI vers Ollama
  
- `convert_embedding_request(request: EmbeddingRequest) -> OllamaEmbeddingRequest`
  - Convertit une requête d'embedding OpenAI vers Ollama
  
- `convert_vision_request(request: VisionRequest) -> OllamaChatRequest`
  - Convertit une requête vision OpenAI vers chat Ollama
  
- `extract_images_from_messages(messages: List[Message]) -> List[str]`
  - Extrait les URLs d'images depuis les messages

### `rkllama.py`
**Rôle** : Conversions spécifiques au format RKLLama (fichier vide/placeholder)

## Architecture
Ces convertisseurs permettent une interopérabilité totale entre les formats d'API :
- Les routes Ollama peuvent recevoir des requêtes et les convertir en format OpenAI interne
- Les routes OpenAI peuvent recevoir des requêtes et les convertir en format Ollama interne
- Le traitement interne utilise un format unifié

## Dépendances
- `core.api.parameters.*` : Modèles Pydantic pour tous les formats
- `typing` : Annotations de types
