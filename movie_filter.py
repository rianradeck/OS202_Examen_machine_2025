# Ce programme va charger n images et y appliquer un filtre de netteté
# puis les sauvegarder dans un dossier de sortie

from PIL import Image
import os
import numpy as np
from scipy import signal
import time

from mpi4py import MPI

globCom = MPI.COMM_WORLD
nbp     = globCom.size
rank    = globCom.rank

# Fonction pour appliquer un filtre de netteté à une image
def apply_filter(image):
    # On charge l'image
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    # Conversion en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy et on normalise
    img = np.repeat(np.repeat(np.array(img), 2, axis=0), 2, axis=1)
    img = np.array(img, dtype=np.double)/255.
    print(f"Nouvelle taille : {img.shape}")
    # Tout d'abord, on crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same')
    # On crée un masque de netteté
    mask = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]])
    # On applique le filtre de netteté
    sharpen_image = np.zeros_like(img)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same'), 0., 1.)

    sharpen_image *= 255.
    sharpen_image = sharpen_image.astype(np.uint8)
    # On retourne l'image modifiée
    return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

start_of_time = time.time()
path = "datas/perroquets/"
# On crée un dossier de sortie
if not os.path.exists("sorties/perroquets"):
    os.makedirs("sorties/perroquets")
out_path = "sorties/perroquets/"

treatment_times = []
output_images = []
rank_range = range(rank, 37, nbp)
print(f"Rank {rank} will treat {len(rank_range)} images: {rank_range}")
for i in rank_range:
    start = time.time()
    image = path + "Perroquet{:04d}.jpg".format(i+1)
    sharpen_image = apply_filter(image)
    # On sauvegarde l'image modifiée
    output_images.append(sharpen_image)
    print(f"Image {i+1} traitée")
    treatment_times.append(time.time() - start)
    print(f"Temps de traitement: {treatment_times[-1]}")
print("Traitement terminé")

if rank == 0:
    print(f"Temps moyen de traitement: {np.mean(treatment_times)}")
    print(f"Temps total de traitement: {np.sum(treatment_times)}")

# On sauvegarde les images modifiées
for i, img in enumerate(output_images):
    img.save(out_path + "Perroquet{:04d}.jpg".format(i+1))
print("Images sauvegardées")

if rank == 0:
    print("Gobal time: ", time.time() - start_of_time)