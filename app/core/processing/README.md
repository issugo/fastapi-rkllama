# Package `app/core/processing`

## Rôle
Orchestre l'exécution des modèles via une architecture multi-processus avec workers isolés pour chaque modèle.

## Fichiers principaux

### `WorkerManager.py`
**Classe** : `WorkerManager`
- **Rôle** : Gestionnaire principal des workers d'inférence
- **Attributs** :
  - `backend_type: BackendType` : Type de backend (RKLLM/RKNN)
  - `workers: Dict[str, Worker]` : Workers actifs indexés par model_id
  - `npu_lock: threading.Lock` : Lock pour accès NPU
  
- **Méthodes principales** :
  - `add_worker(modelfile, full_model_parameters, prompt_cache_path)` : Crée un nouveau worker
  - `exists_model_loaded(model_id) -> bool` : Vérifie si un modèle est chargé
  - `inference(model_id, model_input)` : Lance une inférence
  - `embedding(model_id, model_input)` : Génère des embeddings
  - `multimodal(model_id, prompt_input, images)` : Inférence multimodale
  - `stop_worker(model_id)` : Arrête un worker
  - `stop_all()` : Arrête tous les workers
  - `clear_cache_worker(model_id)` : Vide le cache d'un worker
  - `unload_oldest_models_from_memory(memory_required)` : Libère de la mémoire
  - `is_memory_available_for_model(model_size) -> bool` : Vérifie la mémoire disponible
  - `get_available_base_domain_id(reverse_order) -> int|None` : Alloue un domaine NPU

**Fonction** :
- `get_worker_manager() -> WorkerManager` : Retourne l'instance singleton

### `Worker.py`
**Classe** : `Worker(Process)`
- **Rôle** : Processus worker isolé pour exécuter un modèle
- **Attributs** :
  - `model: Model` : Modèle chargé
  - `backend: Backend` : Backend d'inférence
  - `task_queue: Queue` : Queue de tâches entrantes
  - `result_queue: Queue` : Queue de résultats sortants
  
- **Méthodes** :
  - `run()` : Boucle principale du worker
  - `process_task(task)` : Traite une tâche
  - `handle_inference(input_data)` : Gère l'inférence
  - `handle_embedding(input_data)` : Gère les embeddings
  - `handle_multimodal(prompt, images)` : Gère le multimodal

### `Request.py`
**Classes** :
- `Request(BaseModel)` : Représente une requête d'inférence
  - **Attributs** :
    - `request_id: str` : ID unique
    - `model_id: str` : ID du modèle
    - `input_data: Any` : Données d'entrée
    - `timestamp: datetime` : Horodatage

### `WorkerModelInfo.py`
**Classe** : `WorkerModelInfo(BaseModel)`
- **Rôle** : Informations sur un modèle chargé dans un worker
- **Attributs** :
  - `model_id: str`
  - `base_domain_id: int`
  - `memory_usage: int`
  - `last_used: datetime`

### `Tasks.py`
**Enum** : `TaskType`
- `INFERENCE` : Tâche d'inférence
- `EMBEDDING` : Tâche d'embedding
- `MULTIMODAL` : Tâche multimodale
- `STOP` : Arrêt du worker
- `CLEAR_CACHE` : Vider le cache

### `APIHandler.py`
**Classe** : `APIHandler(ABC)`
- **Rôle** : Classe abstraite pour handlers d'API
- **Méthodes abstraites** :
  - `handle_request(request)` : Traite une requête

### `BaseDomainId.py`
**Type** : `BaseDomainId = int`
- Représente un ID de domaine NPU (1-10)

## Sous-packages

### `api_handlers/`
Handlers spécifiques pour chaque format d'API (Ollama, OpenAI, RKLLama).

### `endpoints/`
Handlers pour les différents types d'endpoints (chat, generate, etc.).

## Architecture
Le système utilise une architecture multi-processus :

```
┌─────────────────┐
│ WorkerManager   │ ← Orchestrateur principal
└────────┬────────┘
         │
    ┌────┴─────────────────┐
    │                      │
┌───▼────┐            ┌───▼────┐
│Worker 1│            │Worker 2│ ← Processus isolés
│Model A │            │Model B │
└────────┘            └────────┘
```

### Avantages :
1. **Isolation** : Chaque modèle dans son propre processus
2. **Parallélisme** : Plusieurs modèles peuvent tourner simultanément
3. **Stabilité** : Un crash de worker n'affecte pas les autres
4. **Gestion mémoire** : Déchargement automatique des modèles inactifs
5. **NPU Domain** : Attribution de domaines NPU distincts

## Dépendances
- `multiprocessing` : Architecture multi-processus
- `threading` : Synchronisation
- `psutil` : Monitoring de mémoire
- `core.backends.*` : Backends d'inférence
- `core.model.*` : Gestion de modèles
