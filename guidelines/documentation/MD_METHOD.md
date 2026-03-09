# Markdown agent process

Les documents en markdown sont rédigés en français.

Chaque package dans /app doit contenir :
- 1. README.md

## 1. README.md
C'est un fichier Markdown qui décrit le contenu du package et ses fonctionnalités.
Il contient, dans l'order :
- 1. **Rôle** : Le rôle applicatif des fichiers contenus dans ce package ;
- 2. **Structure** : la structure de ce qui est contenu dans le package ;
   - La structure du package doit être claire et cohérente pour faciliter la navigation et la compréhension du code. Les fichiers doivent être organisés de manière logique et cohérente avec les fonctionnalités qu'ils implémentent.
- 3. **Fichiers principaux** : Liste des fichiers principaux (ceux qui définissent les fonctionnalités principales du package, ils doivent être documentés concisement et bien nommés pour faciliter la compréhension et la maintenance du code),
   - 3.1. on commence toujours par `__init__.py` (s'il y en a un),
   - 3.2. suivi de chaque fichier python dans ce package, par ordre de nombre de lignes (du plus gros au plus petit).
   - 3.3. ensuite, on liste les fichiers de constantes (s'il y en a),
   - 3.4. ensuite, on liste les sous-package (s'il y en a),
- 4. **Architecture** : description de l'architecture des composants du package ;
- 5. **Dépendances** : liste des dépendances exterbes du package ;
- 6. **Utilisation** : définit comment est exploité le contenu de ce package. 

