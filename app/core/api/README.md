# Package `app/core/api`

## Rôle
Gère la conversion bidirectionnelle entre les différents formats d'API (Ollama ↔ OpenAI) et fournit les modèles Pydantic pour les requêtes/réponses.

## Structure

```
api/
├── conversion/         # Convertisseurs de formats
├── parameters/         # Modèles Pydantic
└── messages_utils.py   # Utilitaires de messages
```

## Sous-packages

### `conversion/`
Contient les classes de conversion entre formats d'API.

### `parameters/`
Définit les modèles Pydantic pour requêtes et réponses de chaque format d'API.

## Fichiers

### `messages_utils.py`
**Fonctions** :
- Utilitaires pour manipuler et formater les messages de chat

## Architecture
Ce package sert de couche d'abstraction permettant aux routes de recevoir des requêtes dans n'importe quel format (Ollama, OpenAI) et de les traiter uniformément via un format interne commun.
