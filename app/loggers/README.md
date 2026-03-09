# Package `app/loggers`

## Rôle
Système de logging centralisé pour l'application avec support de multiples formatters et destinations.

## Fichiers

### `logger.py`
**Fonction** : `logging_setup(log_dir: Path, debug_mode: bool)`
- **Rôle** : Configure le système de logging global
- **Paramètres** :
  - `log_dir` : Répertoire pour les fichiers de logs
  - `debug_mode` : Active le mode debug
  
- **Configuration** :
  - Handlers console et fichier
  - Rotation des logs
  - Niveaux de log configurables
  - Formatters personnalisés

### `loggerConsoleFormatter.py`
**Classe** : `ConsoleFormatter(logging.Formatter)`
- **Rôle** : Formateur coloré pour la console
- **Features** :
  - Coloration selon niveau (ERROR=rouge, WARNING=jaune, INFO=bleu, DEBUG=gris)
  - Format : `[TIMESTAMP] [LEVEL] [MODULE] Message`
  - Support ANSI colors

### `loggerRawFormatter.py`
**Classe** : `RawFormatter(logging.Formatter)`
- **Rôle** : Formateur simple pour fichiers
- **Format** : `TIMESTAMP - LEVEL - MODULE - Message`
- Sans couleurs, optimisé pour parsing

### `logIncomingRequest.py`
**Middleware** : `LogIncomingRequest`
- **Rôle** : Middleware FastAPI pour logger les requêtes entrantes
- **Logs** :
  - Méthode HTTP
  - Path
  - Query parameters
  - Headers (sélectionnés)
  - Body (si applicable)
  - Durée de traitement

### `StreamDebugger.py`
**Classe** : `StreamDebugger`
- **Rôle** : Helper pour déboguer les streams SSE
- **Méthodes** :
  - `log_chunk(chunk)` : Log un chunk de stream
  - `validate_sse(data)` : Valide le format SSE

### `debug_utils.py`
**Fonctions** :
- `add_debug_api(app: FastAPI)` : Ajoute des endpoints de debug
  - `GET /debug/config` : Affiche la configuration
  - `GET /debug/workers` : Affiche l'état des workers
  - `GET /debug/models` : Affiche les modèles chargés
  - `GET /debug/memory` : Affiche l'utilisation mémoire

## Configuration
Le système de logging crée :
- **Console handler** : Logs à l'écran (INFO et supérieur)
- **File handler** : `logs/rkllama.log` (tous les niveaux)
- **Error file handler** : `logs/error.log` (ERROR uniquement)
- **Rotation** : 10 MB max par fichier, 5 fichiers de backup

## Niveaux de log
- **DEBUG** : Informations détaillées de développement
- **INFO** : Événements normaux de l'application
- **WARNING** : Situations inhabituelles mais gérables
- **ERROR** : Erreurs nécessitant attention
- **CRITICAL** : Erreurs graves empêchant l'exécution

## Loggers disponibles
- `rkllama.server` : Serveur principal
- `rkllama.worker_manager` : Gestion des workers
- `core.rkllm.callback` : Callbacks RKLLM
- `api` : Routes API
- Etc.

## Utilisation
```python
import logging

logger = logging.getLogger("rkllama.your_module")
logger.info("Message informatif")
logger.warning("Attention!")
logger.error("Erreur!", exc_info=True)
```

## Mode Debug
Quand activé via configuration :
- Level DEBUG activé
- Logs plus verbeux
- Endpoints de debug disponibles
- Validation accrue

## Dépendances
- `logging` : Module standard Python
- `colorama` : Couleurs console (optionnel)
- `fastapi` : Pour middleware de logging
