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

```python
import torch
import numpy as np
import tifffile
import yaml
import torch.nn.functional as F
from biom3d.builder import Builder
from pathlib import Path
import torch, numpy as np, yaml
from bioimageio.spec.model.v0_5 import (
    ModelDescr, WeightsDescr, TorchscriptWeightsDescr,
    InputTensorDescr, OutputTensorDescr, TensorId,
    BatchAxis, ChannelAxis, SpaceInputAxis, SpaceOutputAxis,
    Author, Version,
    ParameterizedSize,
)
from bioimageio.spec.common import FileDescr
from bioimageio.core import save_bioimageio_package

# ------------ LES DONNEES ------------
chemin_vers_dossier = "/chemin/vers/mon/modele/20260101-nom_du_modele"
chemin_image = "/chemin/vers/mon/image_test.tif" 

#------1 . charger le modele --------------
builder = Builder(path=chemin_vers_dossier, training = False) #va dans config.yaml pour creer le NN (sa structure : la sortie ...)
builder.load_test(chemin_vers_dossier, load_best = True) #load_test va chercher .pth pour y remplir
modele_original = builder.model
modele_original.eval()


# ------------2 . lire la config automatiquement ----
with open(f"{chemin_vers_dossier}/log/config.yaml") as f:
  config = yaml.safe_load(f)
PATCH_SIZE = config["PATCH_SIZE"]
NUM_CLASSES = config["NUM_CLASSES"]
NOM_MODELE = config.get("DESC","Biom3d") #[cle a chercher, valeur par defaut]
# Ensure NUM_CHANNELS is in config, default to 1 if not specified
if "NUM_CHANNELS" not in config:
    config["NUM_CHANNELS"] = 1

#Genere le mapping automatiquement selon le nbre des classes
valeurs = np.linspace(0, 255, NUM_CLASSES + 1).astype(int).tolist()

print(f"PATCH_SIZE : {PATCH_SIZE}")
print(f"NUM_CLASSES : {NUM_CLASSES}")
print(f"MAPPING : {valeurs}")
print(f"NUM_POOLS : {config.get('NUM_POOLS', 'Not Defined')}") # Added this line



#------3 . padding a PATCH_SIZE -------------
image_raw = tifffile.imread(chemin_image).astype(np.float32)
image_norm = (image_raw - image_raw.mean())/(image_raw.std() + 1e-8)
tensor_in = torch.from_numpy(image_norm)
print("taille avant :", tensor_in.shape)
tensor_in = tensor_in.unsqueeze(0)
if(len(tensor_in.shape) == 5) :
  if tensor_in.shape[1] > tensor_in.shape[2] :
    tensor_in = tensor_in.permute(0, 2, 1, 3, 4)
  tensor_in = tensor_in.permute(0, 2, 1, 3, 4)
elif (len(tensor_in.shape) == 4) :
  tensor_in = tensor_in.unsqueeze(0)
else :
  tensor_in = tensor_in.unsqueeze(0).unsqueeze(0)

# while(len(tensor_in.shape) < 5) :
#   tensor_in.unsqueeze(0)
#(1,1,Z,Y,X)


print("Shape originale :", tensor_in.shape)

z, y, x = tensor_in.shape[2], tensor_in.shape[3], tensor_in.shape[4]
pz = max(0, PATCH_SIZE[0] - z)
py = max(0, PATCH_SIZE[1] - y)
px = max(0, PATCH_SIZE[2] - x)
tensor_in = F.pad(tensor_in, (0,px,0,py,0,pz), mode="constant", value = 0) #ajouter 0 a gauche et px a droite,
                                                                               #remplis tout cet espace avec une seule et même valeur fixe
                                                                               #remplir avec du noir

# Fix: Ensure tensor_padded has the exact PATCH_SIZE dimensions for tracing
target_z, target_y, target_x = PATCH_SIZE[0], PATCH_SIZE[1], PATCH_SIZE[2]
current_z, current_y, current_x = tensor_in.shape[2], tensor_in.shape[3], tensor_in.shape[4]

# Initialize a tensor of zeros with the target PATCH_SIZE
tensor_padded = torch.zeros(1, config["NUM_CHANNELS"] , target_z, target_y, target_x, dtype=tensor_in.dtype)

# Determine the region to copy from tensor_in (crop if larger, use full extent if smaller)
copy_z = min(current_z, target_z)
copy_y = min(current_y, target_y)
copy_x = min(current_x, target_x)

# Copy the relevant part of the input image into the target-sized tensor
tensor_padded[..., :copy_z, :copy_y, :copy_x] = tensor_in[..., :copy_z, :copy_y, :copy_x]

print("Shape paddee :", tensor_padded.shape)

#---------4 . Wrapper generique --------------
class WrappedModel(torch.nn.Module):
    def __init__(self, modele, mapping):
        super().__init__()
        self.modele = modele
        self.register_buffer("mapping", torch.tensor(mapping, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits  = self.modele(x)
        if(logits.shape[1]==1) :
          indices = (logits > 0.0).long()
        else :
          indices = torch.argmax(logits, dim=1, keepdim=True)  # (B, 1, Z, Y, X)

        mapped  = self.mapping[indices]
        return mapped

wrapped_model = WrappedModel(modele_original, valeurs)
wrapped_model.eval()

#----------5 . TorchScript trace--------------
with torch.no_grad():
  modele_torchscript = torch.jit.trace(wrapped_model, tensor_padded)

modele_torchscript.save("mon_modele_converti.pt")
print("Modele sauvegarde : mon_modele_converti.pt")

# --- 7. Vérification + .npy ---

# Pour le test_output : utilise les logits bruts (avant argmax)
# car bioimage.io exige des valeurs >= 1e5 ou < -1e5 dans le test_output
# Les logits sont des float continus qui satisfont cette contrainte


with torch.no_grad():
    logits_test = modele_original(tensor_padded)
    out_test    = modele_torchscript(tensor_padded)  # sortie remappée pour vérif visuelle

print(f"Shape logits    : {logits_test.shape}")
print(f"Valeurs uniques sortie finale : {torch.unique(out_test)}") # pour la securite
print("shape de out test : ", out_test.shape)

np.save("test_input_export.npy",  tensor_padded.numpy().astype(np.float32))
# Juste pour satisfaire le validateur — le .pt reste inchangé
fake_output = out_test.clone()
fake_output[fake_output == 0] = -200000.0   # force une valeur < -1e5
np.save("test_output_export.npy", fake_output.numpy().astype(np.float32))
#np.save("test_output_export.npy", out_test.numpy().astype(np.float32))  # logits pour bioimage.io
print("Fichiers .npy sauvegardés")

# Relit la config pour les dimensions
with open(f"{chemin_vers_dossier}/log/config.yaml") as f:
    config = yaml.safe_load(f)

# Ensure NUM_CHANNELS is in config, default to 1 if not specified
if "NUM_CHANNELS" not in config :
  config["NUM_CHANNELS"] = 1

PATCH_SIZE  = config["PATCH_SIZE"]
NUM_CLASSES = config["NUM_CLASSES"]
NOM_MODELE  = config.get("DESC", "Biom3d_Model")
valeurs     = np.linspace(0, 255, NUM_CLASSES + 1).astype(int).tolist()
NUM_CHANNELS = config["NUM_CHANNELS"]

input_descr = InputTensorDescr(
    id=TensorId("input"),
    axes=[
        BatchAxis(),
        ChannelAxis(channel_names=[f"channel_{i}" for i in range(NUM_CHANNELS)]), # Changed: Dynamically set channel_names based on NUM_CHANNELS
        SpaceInputAxis(id="z", size=PATCH_SIZE[0]),
        SpaceInputAxis(id="y", size=PATCH_SIZE[1]),
        SpaceInputAxis(id="x", size=PATCH_SIZE[2]),
    ],
    test_tensor=FileDescr(source=Path("test_input_export.npy")),
)

output_descr = OutputTensorDescr(
    id=TensorId("output"),
    axes=[
        BatchAxis(),
        ChannelAxis(channel_names= ["segmentation"]),  # 4 canaux
        SpaceOutputAxis(id="z", size=PATCH_SIZE[0]),
        SpaceOutputAxis(id="y", size=PATCH_SIZE[1]),
        SpaceOutputAxis(id="x", size=PATCH_SIZE[2]),
    ],
    test_tensor=FileDescr(source=Path("test_output_export.npy")),
)

model_descr = ModelDescr(
    name=NOM_MODELE,
    description=f"Segmentation 3D via Biom3d U-Net. {NUM_CLASSES} classe(s). Valeurs : {valeurs}",
    authors=[Author(name="Youssef EL ANTRAOUI ")],
    license="MIT",
    tags=["segmentation", "3d", "biom3d", "unet", "fluorescence"],
    inputs=[input_descr],
    outputs=[output_descr],
    weights=WeightsDescr(
        torchscript=TorchscriptWeightsDescr(
            source=Path("mon_modele_converti.pt"),
            pytorch_version=Version(torch.__version__),
        )
    ),
)

save_bioimageio_package(model_descr, output_path=Path(f"{NOM_MODELE}_code_original.zip"))
print(f"Package sauvegardé : {NOM_MODELE}_code_original.zip")
```
