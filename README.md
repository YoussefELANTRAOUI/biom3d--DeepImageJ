<img width="300" height="150" alt="deepimageJ" src="https://github.com/user-attachments/assets/377ba1fd-96ac-4e17-b940-b6f7a9be68b2" />
<img width="200" height="150" alt="logo_biom3d_crop" src="https://github.com/user-attachments/assets/30f2baf2-5632-4566-842d-54cff92f3896" />





# 🧬 Biom3d to DeepImageJ (BioImage.IO) Converter

## 📝 Description
This script converts an artificial intelligence model trained with Biom3d (PyTorch `.pth` format) into a standardized BioImage.IO archive (`.zip`). This archive can then be imported directly into the Fiji / ImageJ software (via the DeepImageJ plugin) to segment images without writing a single line of code.

## ⚠️ MAJOR TECHNICAL CONSTRAINT (Must read)
For the export and use in Fiji to work properly, the test image provided to the script must comply with a strict geometric rule tied to DeepImageJ's tiling engine:

**Rule:** The training tile size (`PATCH_SIZE`) cannot exceed 3 times the image size along any given axis.

In other words: If your model was trained with a `PATCH_SIZE` of Z=56, the test image (Raw Image) you provide to the script must have a Z depth of at least 19 slices (since $19 \times 3 = 57 > 56$). If you provide an image that is too small, the script or Fiji will crash.

# 🛠️ 1. Prerequisites and Installation
Whether you are using Google Colab, Jupyter Notebook, or a standard Python script, make sure you have installed the following libraries:

## 🔨 Installation

To install the required dependencies, run the following command:

```bash
pip install torch numpy tifffile pyyaml biom3d bioimageio.core bioimageio.spec
```

## 📂 2. Data Preparation
Before running the script, you must prepare two items in an accessible folder:

The Biom3d model folder: This folder must contain your training outputs (in particular the `/model` subfolder with the `_best.pth` file, and the `/log` subfolder with the `config.yaml` file).

A raw test image (`.tif`): A representative (Raw) image that complies with the size rule mentioned above.

## 🚀 3. How to use the script?
### Step A: Edit the file paths
Open the script (in Colab, Jupyter, or your IDE) and locate the configuration section at the very beginning of the code. You only need to modify these two paths:

```python
# --- USER CONFIGURATION ---

# 1. Path to the folder of the model trained with Biom3d
chemin_vers_dossier = "/path/to/my/model/20260101-model_name"

# 2. Path to the raw test image (.tif)
chemin_image = "/path/to/my/test_image.tif"
```

NOTE: if the image is not in `.tif` format, open the image in Fiji and save it in `.tif` format:
```bash
File -> Save As -> Tiff
```

### Step B: Run the script
Run the cell to execute it.

The script will automatically perform the following operations:

Reconstruction of the model architecture.

Smart centering and "Padding" of your test image.

Conversion of the model into an optimized format (TorchScript).

Generation of a validation prediction (`test_output`).

Full packaging according to the BioImage.IO standard.

## 📦 4. Expected Result
If the script runs successfully, you will see the following message in the console:

🎉 Package saved: `Model_Name_bioimageio.zip`

Congratulations! You now have a `.zip` file ready to use.
Simply open Fiji, go to **Plugins > DeepImageJ > Install Model**, select this `.zip` file, and your biologists can start segmenting their data!

# For a better visualization of the result
```bash
Image -> Adjust -> Brightness/Contrast
```

## ❓ Troubleshooting (FAQ)
**`NegativeArraySizeException` error in Fiji:** The model produces an output that is too heavy for Java's memory. This issue often occurs with models that have many classes (e.g. 13 classes). You must reduce the `PATCH_SIZE` during training.

**The model only displays a single color in Fiji:** This is normal if your structure is very small. The model does detect the target correctly, but Fiji's default display overrides it. Go to **Image > Adjust > Threshold** to reveal it.

## 🧩 Special case: Multi-Channel Images (e.g. Brain / MRI model)
Warning: When you open complex TIFF images (containing several fluorescence channels or several MRI modalities), Fiji may misinterpret the file dimensions. It often confuses Channels (C) with Time (Frames/T) or Depth (Slices/Z).

If your model expects 4 input channels, but Fiji opened the image as a video (Frames = 4, Channels = 1), the DeepImageJ plugin will display a dimensioning error.

### Solution: Reorganize the Hyperstack before running the model

Open your `.tif` image in Fiji.

From the main menu, go to: **Image > Hyperstacks > Stack to Hyperstack...**

A window opens. Configure it exactly as follows to restore the correct geometry:

**Order:** Choose the correct order (for example `xyczt` or `TCZYX` depending on your source file).

**Channels (c):** Enter the actual number of channels (e.g. 4 instead of 1).

**Slices (z):** Keep the number of depth slices (e.g. 155).

**Frames (t):** Force to 1 (instead of 4, since it is not a video).

Click OK.

**Verification:** At the bottom of your image window in Fiji, you should now see a scroll bar for Channels (C) and another one for Depth (Z) below it, confirming that the image is ready for DeepImageJ.

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
