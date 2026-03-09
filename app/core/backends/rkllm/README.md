# Package `app/core/backends/rkllm`

## Rôle
Implémente le backend RKLLM pour l'inférence de modèles LLM sur processeurs Rockchip RK3588/RK3576.

## Fichiers

### `rkllm_backend.py`
**Classe** : `RKLLMBackend(Backend)`
- **Rôle** : Backend principal pour l'exécution de modèles RKLLM
- **Attributs** :
  - `model: Model` : Instance du modèle
  - `options: FullModelParameters` : Paramètres de génération
  - `base_domain_id: BaseDomainId` : ID de domaine NPU
  - `handle` : Handle de la bibliothèque RKLLM
  - `rkllm_param` : Paramètres RKLLM natifs
  - `lora_adapter_path` : Chemin vers l'adaptateur LoRA (optionnel)
  
- **Méthodes** :
  - `__init__(model, options, base_domain_id, prompt_cache_path, lora_model_path)` : Initialise le backend et charge le modèle
  - `tokens_to_ctypes_array(tokens, ctype)` : Convertit des tokens en tableau C
  - `set_function_tools(system_prompt, tools, tool_response_str)` : Configure les function tools
  - `run(prompt_tokens)` : Exécute l'inférence sur les tokens d'entrée
  - `abort()` : Annule l'inférence en cours
  - `clear_cache()` : Vide le cache KV
  - `release()` : Libère les ressources RKLLM

### `classes.py`
**Classes de structures C/ctypes** :

- `RKLLMExtendParam(ctypes.Structure)` : Paramètres étendus RKLLM
  - `base_domain_id` : ID du domaine de base
  - `npu_num` : Nombre de NPUs à utiliser

- `RKLLMLoraAdapter(ctypes.Structure)` : Configuration adaptateur LoRA
  - `lora_adapter_path` : Chemin vers l'adaptateur
  - `lora_adapter_name` : Nom de l'adaptateur
  - `scale` : Facteur d'échelle

- `RKLLMInferParam(ctypes.Structure)` : Paramètres d'inférence
  - `mode` : Mode d'inférence
  - `lora_params` : Paramètres LoRA (optionnel)

- `RKLLMParam(ctypes.Structure)` : Paramètres principaux du modèle
  - `model_path` : Chemin du fichier modèle
  - `max_context_len` : Longueur maximale du contexte
  - `max_new_tokens` : Nombre max de nouveaux tokens
  - `top_k`, `top_p`, `temperature` : Paramètres de sampling
  - `repeat_penalty` : Pénalité de répétition
  - `frequency_penalty`, `presence_penalty` : Pénalités OpenAI
  - `mirostat`, `mirostat_tau`, `mirostat_eta` : Paramètres Mirostat
  - `skip_special_token` : Ignorer les tokens spéciaux
  - `is_async` : Mode asynchrone
  - `img_start`, `img_end`, `img_content` : Tokens d'image
  - `extend_param` : Paramètres étendus

- `RKLLMResultLastHiddenLayer(ctypes.Structure)` : Dernière couche cachée

- `RKLLMTokenInfo(ctypes.Structure)` : Informations sur un token
  - `token_id` : ID du token
  - `logprob` : Log probabilité
  - `top_logprobs` : Top log probabilités

- `RKLLMResult(ctypes.Structure)` : Résultat de l'inférence
  - `text` : Texte généré
  - `token_id` : ID du token
  - `num_tokens` : Nombre de tokens
  - `tokens_info` : Informations détaillées sur les tokens

### `callback.py`
**Fonction** : `callback_impl(result, userdata, state)`
- **Rôle** : Fonction callback appelée par la bibliothèque RKLLM pour chaque token généré
- **Paramètres** :
  - `result: RKLLMResult` : Résultat de génération
  - `userdata` : Données utilisateur (queue de résultats)
  - `state` : État de la génération (0=en cours, 1=terminé, <0=erreur)
- **Comportement** :
  - Met les tokens générés dans la queue
  - Gère les erreurs et l'état de fin

## Architecture
Le backend RKLLM utilise `ctypes` pour interagir avec la bibliothèque native C de Rockchip :
1. Charge le modèle RKLLM via `rkllmInit()`
2. Configure les paramètres d'inférence
3. Exécute l'inférence de manière asynchrone avec callbacks
4. Reçoit les tokens générés via la fonction callback
5. Gère le cache KV pour optimiser les performances

## Dépendances
- `ctypes` : Interface avec bibliothèque C
- `multiprocessing` : Queues pour communication inter-processus
- Bibliothèque native `librkllmrt.so` : Runtime RKLLM de Rockchip
- `core.model.Model` : Gestion de modèles
- `core.processing.BaseDomainId` : Allocation de domaines NPU
