import os
import torch
from collections import namedtuple
from PIL import Image

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Utilisation du périphérique :", device)

# Classe INFRA10Class
INFRA10Class = namedtuple('CityscapesClass', ['name', 'train_id', 'category', 'category_id',
                                              'has_instances', 'ignore_in_eval', 'color', 'grey'])

# Définition des classes
classes = [
    INFRA10Class('road',                 0, 'flat', 1, False, False, (224, 92, 94), (0, 0, 0)),
    INFRA10Class('sidewalk',             1, 'flat', 1, False, False, (98, 229, 212), (10, 10, 10)),
    INFRA10Class('building',             2, 'construction', 2, False, False, (75, 213, 234), (20, 20, 20)),
    INFRA10Class('wall',                 3, 'construction', 2, False, False, (42, 186, 83), (30, 30, 30)),
    INFRA10Class('fence',                4, 'construction', 2, False, False, (65, 255, 12), (40, 40, 40)),
    INFRA10Class('pole',                 5, 'object', 3, False, False, (46, 181, 211), (50, 50, 50)),
    INFRA10Class('traffic light',        6, 'object', 3, False, False, (38, 173, 42), (60, 60, 60)),
    INFRA10Class('traffic sign',         7, 'object', 3, False, False, (237, 61, 222), (70, 70, 70)),
    INFRA10Class('vegetation',           8, 'nature', 4, False, False, (122, 234, 2), (80, 80, 80)),
    INFRA10Class('terrain',              9, 'nature', 4, False, False, (86, 244, 247), (90, 90, 90)),
    INFRA10Class('sky',                  10, 'sky', 5, False, False, (87, 242, 87), (100, 100, 100)),
    INFRA10Class('person',               11, 'human', 6, True, False, (33, 188, 119), (110, 110, 110)),
    INFRA10Class('rider',                12, 'human', 6, True, False, (216, 36, 186), (120, 120, 120)),
    INFRA10Class('car',                  13, 'vehicle', 7, True, False, (224, 172, 51), (130, 130, 130)),
    INFRA10Class('truck',                14, 'vehicle', 7, True, False, (232, 196, 97), (140, 140, 140)),
    INFRA10Class('bus',                  15, 'vehicle', 7, True, False, (0, 137, 150), (150, 150, 150)),
    INFRA10Class('train',                16, 'vehicle', 7, True, False, (97, 232, 187), (160, 160, 160)),
    INFRA10Class('motorcycle',           17, 'vehicle', 7, True, False, (239, 107, 197), (170, 170, 170)),
    INFRA10Class('bicycle',              18, 'vehicle', 7, True, False, (149, 15, 252), (180, 180, 180)),
    INFRA10Class('unlabeled',            255, 'void', 0, False, True, (206, 140, 26), (255, 255, 255)),
]

def convert_images_to_greyscale(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste pour stocker les images en niveaux de gris
    grayscale_images = []

    # Parcourir les fichiers du dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") and not filename.endswith("labelIds.png"):
            # Chemin complet des fichiers d'entrée et de sortie
            input_path = os.path.join(input_folder, filename)

            # Charger l'image et la convertir en mode RVB
            image = Image.open(input_path)
            image = image.convert("RGB")

            grayscale_images.append(image)

    # Convertir toutes les images en niveaux de gris en une seule opération batch
    batch_images = torch.stack([torch.Tensor(list(img.getdata())).view(img.size[1], img.size[0], 3)
                                for img in grayscale_images]).to(device)

    for i in range(len(classes)):
        color = torch.Tensor(classes[i].color).view(1, 1, 1, 3)
        color = color.to(device)

        mask = torch.all(batch_images == color, dim=3)
        batch_images[mask] = torch.Tensor(classes[i].grey).view(1, 1, 1, 3)

    # Convertir les pixels en niveaux de gris en une image PIL
    batch_images = batch_images.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')
    grayscale_images = [Image.fromarray(img) for img in batch_images]

    # Sauvegarder les images modifiées
    for i in range(len(grayscale_images)):
        output_path = os.path.join(output_folder, f"output_{i}.png")
        grayscale_images[i].save(output_path)

# Exemple d'utilisation
input_folder = "microdatabase"
output_folder = "niveaugris"
convert_images_to_greyscale(input_folder, output_folder)
