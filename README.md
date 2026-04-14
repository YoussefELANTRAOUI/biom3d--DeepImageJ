<img width="250" height="250" alt="logo_biom3d_crop" src="https://github.com/user-attachments/assets/0a809224-f931-471d-9cd6-ec36f2e20a67" />

<img width="250" height="250" alt="deepimageJ" src="https://github.com/user-attachments/assets/5aa23e78-4876-4269-9b1e-ca3eec586858" />


# 🧬 Convertisseur Biom3d vers DeepImageJ (BioImage.IO)
## 📝 Description
Ce script permet de convertir un modèle d'intelligence artificielle entraîné avec Biom3d (format PyTorch .pth) en une archive standardisée BioImage.IO (.zip).Cette archive peut ensuite être importée directement dans le logiciel Fiji / ImageJ (via le plugin DeepImageJ) pour segmenter des images sans aucune ligne de code.
## ⚠️ CONTRAINTE TECHNIQUE MAJEURE (À lire absolument)
Pour que l'exportation et l'utilisation dans Fiji fonctionnent, l'image de test fournie au script doit respecter une règle géométrique stricte liée au moteur de tuilage (tiling) de DeepImageJ :Règle : La taille de la tuile d'entraînement (PATCH_SIZE) ne peut pas excéder 3 fois la taille de l'image sur un axe donné.En clair : Si votre modèle a été entraîné avec un PATCH_SIZE de Z=56, l'image de test (Raw Image) que vous donnez au script doit avoir une profondeur Z d'au moins 19 coupes (car $19 \times 3 = 57 > 56$). Si vous fournissez une image trop petite, le script ou Fiji plantera.
# 🛠️ 1. Prérequis et Installation
Que vous utilisiez Google Colab, Jupyter Notebook ou un script Python classique, assurez-vous d'avoir installé les librairies suivantes :
## 🔨 Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```bash
pip install torch numpy tifffile pyyaml biom3d bioimageio.core bioimageio.spec
```
## 📂 2. Préparation des données
Avant de lancer le script, vous devez préparer deux éléments dans un dossier accessible :

Le dossier du modèle Biom3d : Ce dossier doit contenir vos résultats d'entraînement (notamment le sous-dossier /model avec le fichier _best.pth et le sous-dossier /log avec le fichier config.yaml).

Une image brute de test (.tif) : Une image représentative (Raw) qui respecte la règle de taille mentionnée ci-dessus.

## 🚀 3. Comment utiliser le script ?
Étape A : Modifier les chemins d'accès
Ouvrez le script (dans Colab, Jupyter ou votre IDE) et repérez la section de configuration au tout début du code. Vous devez modifier uniquement ces deux chemins :
``` Python
# --- CONFIGURATION UTILISATEUR ---

# 1. Chemin vers le dossier du modèle entraîné par Biom3d
chemin_vers_dossier = "/chemin/vers/mon/modele/20260101-nom_du_modele"

# 2. Chemin vers l'image brute de test (.tif)
chemin_image = "/chemin/vers/mon/image_test.tif"
```
PRECISION: si l'image n'est pas en format .tif, ouvres l'image dans Fiji et sauvegarder en format .tif 
```bash
File -> Save As -> Tiff
```
Étape B : Exécuter le script
Lancez l'exécution de la cellule .

Le script effectuera automatiquement les opérations suivantes :

Reconstruction de l'architecture du modèle.

Centrage intelligent et "Padding" de votre image de test.

Conversion du modèle en format optimisé (TorchScript).

Création d'une prédiction de validation (test_output).

Empaquetage complet selon la norme BioImage.IO.
## 📦 4. Résultat attendu
Si le script s'exécute avec succès, vous verrez le message suivant dans la console :
🎉 Package sauvegardé : Nom_Du_Modele_bioimageio.zip

Félicitations ! Vous avez maintenant un fichier .zip prêt à l'emploi.
Il vous suffit d'ouvrir Fiji, d'aller dans Plugins > DeepImageJ > Install Model, de sélectionner ce fichier .zip, et vos biologistes peuvent commencer à segmenter leurs données !
# Pour bien visualiser le resultat 
```bash
Image->Adjust->Brightness/Contrast
```
## ❓ Dépannage (FAQ)
Erreur NegativeArraySizeException dans Fiji : Le modèle produit une sortie trop lourde pour la mémoire de Java. Ce problème survient souvent sur des modèles avec beaucoup de classes (ex: 13 classes). Il faut réduire le PATCH_SIZE lors de l'entraînement.

Le modèle n'affiche qu'une seule couleur dans Fiji : C'est normal si votre structure est très petite. Le modèle détecte bien la cible, mais l'affichage par défaut de Fiji l'écrase. Faites Image > Adjust > Threshold pour la révéler.
## 🧩 Cas particulier : Images Multi-Canaux (Ex: Modèle Cerveau / IRM)
Attention : Lorsque vous ouvrez des images TIFF complexes (comportant plusieurs canaux de fluorescence ou plusieurs modalités IRM), Fiji peut mal interpréter les dimensions du fichier. Il confond souvent les Canaux (C) avec le Temps (Frames/T) ou la Profondeur (Slices/Z).

Si votre modèle attend 4 canaux en entrée, mais que Fiji a ouvert l'image comme une vidéo (Frames = 4, Channels = 1), le plugin DeepImageJ affichera une erreur de dimensionnement.

Solution : Réorganiser l'Hyperstack avant de lancer le modèle

Ouvrez votre image .tif dans Fiji.

Dans le menu principal, allez dans : Image > Hyperstacks > Stack to Hyperstack...

Une fenêtre s'ouvre. Configurez-la exactement comme suit pour rétablir la bonne géométrie :

Order : Choisissez l'ordre correct (par exemple xyczt ou TCZYX selon votre fichier d'origine).

Channels (c) : Entrez le vrai nombre de canaux (ex: 4 au lieu de 1).

Slices (z) : Laissez le nombre de coupes de profondeur (ex: 155).

Frames (t) : Forcez à 1 (au lieu de 4, car ce n'est pas une vidéo).

Cliquez sur OK.

Vérification : En bas de la fenêtre de votre image dans Fiji, vous devriez maintenant voir une barre de défilement pour les Canaux (C) et une barre pour la Profondeur (Z) en dessous, confirmant que l'image est prête pour DeepImageJ.
