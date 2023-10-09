# Non local maxima suppression (final)

from skimage.feature import peak_local_max
from scipy import ndimage
import numpy as np

import cv2

global count

# from: https://stackoverflow.com/questions/43923648/region-growing-python
def get8n(x, y, shape):
    out = []
    maxx = shape[0]-1
    maxy = shape[1]-1
    
    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    return out

def region_growing(img, out, seed, tol, threshold_abs):
    
    seed_points = []
    seed_points.append((i, j))
    processed = []
    rangeTol = [img[i][j] - tol, img[i][j] + tol]
    
    while (len(seed_points)>0):
        
        pix = seed_points[0]
        out[pix[0]][pix[1]] = 1
        
        for coord in get8n(pix[0], pix[1], img.shape):
            
            if img[coord[0]][coord[1]] > rangeTol[0] and img[coord[0]][coord[1]] < rangeTol[1] and img[coord[0]][coord[1]]>threshold_abs:
                
                out[coord[0]][coord[1]] = 1
                
                if not coord in processed:
                    seed_points.append(coord)
                    
                processed.append(coord)
                
        seed_points.pop(0)
    
    return out

fig, axs = plt.subplots(1, 3, figsize=(8, 8))

prediction_gray = (prediction - np.amin(prediction)) / (np.amax(prediction) - np.amin(prediction)) * 255

out_cell = np.array([[0 for i in range(prediction.shape[1])] for j in range(prediction.shape[0])])
out_guttae = np.array([[0 for i in range(prediction.shape[1])] for j in range(prediction.shape[0])])

cellbound_gray = (cellbound - np.amin(prediction)) / (np.amax(prediction) - np.amin(prediction)) * 255
guttaebound_gray = (guttaebound - np.amin(prediction)) / (np.amax(prediction) - np.amin(prediction)) * 255

localMax_cell = peak_local_max(prediction_gray, indices=False, exclude_border=False, threshold_abs=cellbound_gray)
localMax_guttae = peak_local_max(prediction_gray + 255, indices=False, exclude_border=False, threshold_abs=guttaebound_gray)

count_cell = 0
tol_cell = 70

count_guttae = 0
tol_guttae = 70


plt.figure(figsize=(10,20))

for i in range(localMax_cell.shape[0]):        # rows
    for j in range(localMax_cell.shape[1]):    # cols

        if localMax_cell[i][j]:

            out_cell = region_growing(prediction_gray, out_cell, (i,j), tol_cell, cellbound_gray)
            count_cell += 1
        
        if localMax_guttae[i][j]:

            out_guttae = region_growing(prediction_gray, out_guttae, (i,j), tol_guttae, guttaebound_gray)
            count_guttae += 1

axs[0].imshow(prediction_gray, cmap='gray')
axs[0].axis("off")

axs[1].imshow(out_cell, cmap='gray')
axs[1].axis("off")

axs[2].imshow(out_guttae, cmap='gray')
axs[2].axis("off")