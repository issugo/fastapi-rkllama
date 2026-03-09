# Package `app/api/routes/converter`

## Rôle
Expose des endpoints pour la conversion de modèles depuis différents formats vers le format RKLLM.

## Fichiers

### `rkllm_converter.py`
**Route** : Router FastAPI pour la conversion de modèles

**Endpoints** :
- `convert_rkllm()` : `POST /convert/rkllm`
  - **Paramètres** : `ConversionConfig` - Configuration de conversion
  - **Rôle** : Convertit un modèle HuggingFace vers le format RKLLM
  - **Retour** : 
    - `0` : Succès
    - `1` : Erreur

**Dépendances** :
- `core.api.parameters.converter.ConversionConfig` : Modèle de configuration
- `core.model.converter.HuggingFaceToRKLLMConverter` : Convertisseur HF → RKLLM

## Utilisation
```python
# POST /convert/rkllm
{
  "model_name": "Qwen/Qwen-1.8B-Chat",
  "output_path": "/path/to/output",
  "quantization": "w8a8"
}
```

## Architecture
1. Reçoit la configuration de conversion
2. Crée une instance de `HuggingFaceToRKLLMConverter`
3. Lance la conversion
4. Retourne le statut
