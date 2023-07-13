import os
from collections import namedtuple
from PIL import Image

# Classe INFRA10Class
INFRA10Class = namedtuple('CityscapesClass', ['name', 'train_id', 'category', 'category_id',
                                              'has_instances', 'ignore_in_eval', 'color', 'grey'])

# Définition des classes
classes = [
    INFRA10Class('road',                 0, 'flat', 1, False, False, (224, 92, 94), (0, 0, 0)),
    INFRA10Class('sidewalk',             1, 'flat', 1, False, False, (98, 229, 212), (1, 1, 1)),
    INFRA10Class('building',             2, 'construction', 2, False, False, (75, 213, 234), (2, 2, 2)),
    INFRA10Class('wall',                 3, 'construction', 2, False, False, (42, 186, 83), (3, 3, 3)),
    INFRA10Class('fence',                4, 'construction', 2, False, False, (65, 255, 12), (4, 4, 4)),
    INFRA10Class('pole',                 5, 'object', 3, False, False, (46, 181, 211), (5, 5, 5)),
    INFRA10Class('traffic light',        6, 'object', 3, False, False, (38, 173, 42), (6, 6, 6)),
    INFRA10Class('traffic sign',         7, 'object', 3, False, False, (237, 61, 222), (7, 7, 7)),
    INFRA10Class('vegetation',           8, 'nature', 4, False, False, (122, 234, 2), (8, 8, 8)),
    INFRA10Class('terrain',              9, 'nature', 4, False, False, (86, 244, 247), (9, 9, 9)),
    INFRA10Class('sky',                  10, 'sky', 5, False, False, (87, 242, 87), (10, 10, 10)),
    INFRA10Class('person',               11, 'human', 6, True, False, (33, 188, 119), (11, 11, 11)),
    INFRA10Class('rider',                12, 'human', 6, True, False, (216, 36, 186), (12, 12, 12)),
    INFRA10Class('car',                  13, 'vehicle', 7, True, False, (224, 172, 51), (13, 13, 13)),
    INFRA10Class('truck',                14, 'vehicle', 7, True, False, (232, 196, 97), (14, 14, 14)),
    INFRA10Class('bus',                  15, 'vehicle', 7, True, False, (0, 137, 150), (15, 15, 15)),
    INFRA10Class('train',                16, 'vehicle', 7, True, False, (97, 232, 187), (16, 16, 16)),
    INFRA10Class('motorcycle',           17, 'vehicle', 7, True, False, (239, 107, 197), (17, 17, 17)),
    INFRA10Class('bicycle',              18, 'vehicle', 7, True, False, (149, 15, 252), (18, 18, 18)),
    INFRA10Class('unlabeled',            255, 'void', 0, False, True, (206, 140, 26), (255, 255, 255)),
]

def convert_images_to_greyscale(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Parcourir les fichiers du dossier d'entrée
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") and not filename.endswith("labelIds.png"):
            # Chemin complet des fichiers d'entrée et de sortie
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Charger l'image
            image = Image.open(input_path)
            image = image.convert("RGB")  # Convertir en mode RVB

            pixels = image.load()

            width, height = image.size
            for y in range(height):
                for x in range(width):
                    pixel = pixels[x, y]
                    # Trouver la classe correspondante à la couleur du pixel
                    matching_class = next((c for c in classes if c.color == pixel), None)
                    if matching_class is not None:
                        # Remplacer la couleur du pixel par le niveau de gris correspondant
                        pixels[x, y] = matching_class.grey

            # Sauvegarder l'image modifiée au format JPEG
            image.save(output_path, "png")

# Exemple d'utilisation
input_folder = "/home/poc2014/dataset/temp/INFRA10/semantic_segmentation_truth/train/Trappes/"
output_folder = "/home/poc2014/grey_output/g_Trappes"
convert_images_to_greyscale(input_folder, output_folder)
