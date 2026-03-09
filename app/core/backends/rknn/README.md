# Package `app/core/backends/rknn`

## Rôle
Implémente le backend RKNN pour l'exécution de modèles de réseaux de neurones sur processeurs Rockchip.

## Fichiers

### `rknn_backend.py`
**Classe** : `RKNNBackend(Backend)`
- **Rôle** : Backend pour l'exécution de modèles au format RKNN
- **Attributs** :
  - `model: Model` : Instance du modèle
  - `options: FullModelParameters` : Paramètres de génération
  - `base_domain_id: BaseDomainId` : ID de domaine NPU
  - `handle` : Handle de la bibliothèque RKNN
  
- **Méthodes** :
  - `__init__(model, options, base_domain_id)` : Initialise le backend RKNN
  - `run(input_data)` : Exécute l'inférence
  - `release()` : Libère les ressources

### `classes.py`
**Classes de structures C/ctypes pour RKNN** :

- `RKNNExtendParam(ctypes.Structure)` : Paramètres étendus RKNN
  - Configuration spécifique au runtime RKNN

- `RKNNParam(ctypes.Structure)` : Paramètres principaux RKNN
  - `model_path` : Chemin du modèle RKNN
  - `extend_param` : Paramètres étendus

- `RKNNResult(ctypes.Structure)` : Résultat de l'inférence RKNN
  - `output` : Données de sortie
  - `output_size` : Taille de la sortie

## Architecture
Le backend RKNN utilise la bibliothèque native RKNN de Rockchip pour :
1. Charger des modèles de réseaux de neurones optimisés
2. Exécuter l'inférence sur le NPU Rockchip
3. Retourner les résultats bruts

**Différence avec RKLLM** :
- RKNN : Format général pour réseaux de neurones
- RKLLM : Format spécialisé pour Large Language Models

## Dépendances
- `ctypes` : Interface avec bibliothèque C
- Bibliothèque native `librknn_api.so` : Runtime RKNN de Rockchip
- `core.model.Model` : Gestion de modèles
- `core.processing.BaseDomainId` : Allocation de domaines NPU
