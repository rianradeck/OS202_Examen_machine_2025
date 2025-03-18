# Ce programme double la taille d'une image en assayant de ne pas trop pixeliser l'image.

from PIL import Image
import os
import numpy as np
from scipy import signal

import time

from mpi4py import MPI

globCom = MPI.COMM_WORLD
nbp     = globCom.size
rank    = globCom.rank

def convolute(img):
    # On crée un masque de flou gaussien
    mask = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.
    # On applique le filtre de flou
    blur_image = np.zeros_like(img, dtype=np.double)
    for i in range(3):
        blur_image[:,:,i] = signal.convolve2d(img[:,:,i], mask, mode='same', boundary='symm')
    # On crée un masque de netteté
    mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # On applique le filtre de netteté uniquement sur la luminance :
    sharpen_image = np.zeros_like(img, dtype=np.double)
    sharpen_image[:,:,:2] = blur_image[:,:,:2]
    sharpen_image[:,:,2] = np.clip(signal.convolve2d(blur_image[:,:,2], mask, mode='same', boundary='symm'), 0., 1.)
    # On retourne l'image modifiée
    sharpen_image = (255.*sharpen_image).astype(np.uint8)
    return sharpen_image
    # return Image.fromarray(sharpen_image, 'HSV').convert('RGB')

path = "datas/"
image = path+"paysage.jpg"
start = time.time()

local_img = None
if rank == 0:
    doubling_time = time.time()
    # On charge l'image
    img = Image.open(image)
    print(f"Taille originale {img.size}")
    # Convertir la représentation RGB en HSV :
    img = img.convert('HSV')
    # On convertit l'image en tableau numpy
    img = np.array(img, dtype=np.double)
    # On double sa taille et on la normalise
    img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)/255.
    print(f"Nouvelle taille : {img.shape}")
    print(f"Doubling time: {time.time()-doubling_time:.3f}")
    
    local_img = img[:, :int(np.ceil(img.shape[1] / nbp) + 1), :]
    for _rank in range(1, nbp):
        cols_per_process = np.ceil(img.shape[1] / nbp)
        start_col = int(_rank * cols_per_process)
        end_col = int(min((_rank + 1) * cols_per_process, img.shape[1]))
        
        globCom.send(img[:, start_col - 1:end_col + (_rank != nbp - 1), :], dest=_rank, tag=11)

if rank != 0:
    local_img = globCom.recv(source=0, tag=11)
    # print(f"[{rank}] Received image of shape {local_img.shape}")

convolute_time = time.time()
local_img = convolute(local_img)
# removing "ghost columns"
local_img = local_img[:, (rank != 0):(-1 if rank != nbp-1 else len(local_img[1])), :]
if rank == 0:
    print(f"[{rank}] Convolute time: {time.time()-convolute_time:.3f}")
    print(f"[{rank}] Image shape: {local_img.shape}")
# On sauvegarde l'image modifiée
Image.fromarray(local_img, 'HSV').convert('RGB').save(f"sorties/paysage_double_{rank}.jpg")

if rank != 0:
    globCom.send(local_img, dest=0, tag=12)

if rank == 0:
    gather_time = time.time()
    all_images = [local_img]
    for _rank in range(1, nbp):
        _local_image = globCom.recv(source=_rank, tag=12)
        all_images.append(_local_image)
    
    final_image = np.concatenate(all_images, axis=1)
    print(f"Final image shape: {final_image.shape}")
    final_image = Image.fromarray(final_image, 'HSV').convert('RGB')
    final_image.save(f"sorties/paysage_double.jpg")
    print(f"Gather time: {time.time()-gather_time:.3f}")

    print(f"Global time: {time.time()-start:.3f}")
    print("Image sauvegardée")
