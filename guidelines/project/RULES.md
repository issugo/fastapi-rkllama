# Règles générales pour le projet fastapi-rkllama

Règles à appliquer :
- chaque package contient un document README.md,
- chaque fichier python est documenté,
- chaque méthode est documentée,
- chaque méthode commence par un log décrivant son nom et ses paramètres,
- la documentation est maintenue à jour,
- il est interdit de supprimer des logs ou des commentaires dans le code (l'agent peut seulement en ajouter ou les modifier pour les compléter),
- les docstrings doivent être maintenus à jour après chaque modification (créés si nécessaire, mis à jour sinon),
- chaque fois qu'un agent modifie le code de l'application, la modification doit être documentée directement dans le code. En cas de modifications multiples, un résumé de l'ensemble des changements doit être rédigé dans les commentaires,
- le dossier `src` est déprécié et ne doit pas être utilisé pour ajouter du nouveau contenu de projet (les nouveaux fichiers ou le code refactorisé doivent aller dans `app` ou `tests`),
- lorsque des tests avec un modèle réel doivent être effectués, le modèle Hugging Face `MODEL_NAME=dulimov/Qwen3-4B-rk3588-1.2.1-unsloth-16k` et `MODEL_FILE=Qwen3-4B-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm` doit être utilisé,
- pour exécuter pytest et pytest-bdd, uv doit être utilisé (ex. uv run pytest),
- si le package Python d'un script modifié contient un fichier `README.md`, ce fichier markdown doit être mis à jour pour refléter les modifications apportées au script,
- les vérifications de pre-commit doivent être effectuées après toute modification de code effectuée par l'agent (ex. `uv run pre-commit run --all-files`),
- à la fin d'une tâche d'agent, si le code a été modifié, un commit git doit être effectué avec un message de commit correct respectant la spécification Git Conventional Commits (ex. `feat: ...`, `fix: ...`, `refactor: ...`, `docs: ...`, `test: ...`, `chore: ...`).



