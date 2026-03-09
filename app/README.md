# fastapi-rkllama

## Introduction
Le répertoire /app contient les fichiers source python du projet fastapi-rkllama, une application web utilisant FastAPI avec les bibliothèques rkllm et rknn pour fournir une interface (API) utilisateur pour l'interrogation de modèles de langage instancié localement.
L'application propose les API de Ollama et de OpenAI (et RKLLAMA).

## Lexique
- backend : bibliothèques de support matériel au LLM (RKLLM et RKNN),
- model : descriptif du modèle de LLM (tronc commun et spécificités Ollama et HuggingFace), 
- ModelInfo : contient les informations sur les fichiers du modèle (mais pas leur configuration),
- ModelMetadata : contient les configuration du modèle (mais pas les informations sur les fichiers du modèle),
- model supplier : fournisseur de modèle (HuggingFace, Ollama, ...),
- model storage : gestionnaire de stockage des modèles de LLM commun (transforme et/ou adapte les formats Ollama et/ou HuggingFace),
- endpoint (handler) : handler de requête HTTP dédié aux interactions avec le LLM par type de requête à un LLM (comme "chat", "completion", "embeddings", etc) classé selon les principes d'OpenAI (ne gère pas les problématiques de formattage/transtypage, de la responsabilité du api handler),
- api handlers : handler (workflow, formatage/transtypage, gestion des stream, ...) de requête HTTP par famille d'API (API Ollama, API OpenAI, API RKLLAMA) (ne gère pas la partie spécifique au type de requete, de la responsabilité du endpoint handler),
- request : structure porteuse de l'ensemble des informations permettant de réaliser une requête (le modèle, sa configuration, et les données de requête), par famille d'API (Ollama, OpenAI et RKLLAMA), 
- Task : contient le type de tache (inférence, embedding, vision, ...) et la requete (request), 
- worker : traitement des tâches dédiées aux interactions avec le LLM (traitement d'une request) en fonction d'un model instancié sur un backend et exploitant un api handler,
- WorkerManager : ordonnanceur des taches "worker" (gestion par file) qui gère l'exclusion mutuelle des ressources et la gestion des ressources (backend).

## Routes d'API
Les routes d'API sont dans /app/api/routes, et sont classés selon la famille d'API (Ollama, OpenAI et RKLLAMA).

## Types Pydantic des APIs 
Les types Pydantic des APIs sont dans /app/core/api/parameters :
- aucune données ne doit être dupliquée, 
- un tronc commun est mis en place,
- les spécificités sont classés selon le type d'élément (requête ou réponse) et selon la famille d'API (Ollama, OpenAI et RKLLAMA).

## Services

### Téléchargement de modèle
L'API permet le téléchargement de modèle et le stockage local, ainsi que la définition d'une configuration par défaut pour chaque modèle.

### Exploitation d'un modèle
L'application propose, au travers de l'API Ollama et de l'API OpenAI (et RKLLAMA), d'exploiter un modèle (LLM, embedding, vision, ...).
Elle s'appui sur des capacités matérielles (porté par les endpoints).
Chaque demande est transformé en request, puis empilé sous forme de task, pour être ensuite dépilé par le WorkerManager qui envoie la tache à un worker. 
