from skimage.measure import label, regionprops
import numpy as np
from patchify import patchify, unpatchify
import cv2
import matplotlib.pyplot as plt

def segment(cell, guttae, skel):

	seg = 1-skel
	seg_label = label(seg, connectivity=1)

	guttae[seg==0] = 0

	cell_label = label(cell, connectivity=1)
	guttae_label = label(guttae, connectivity=1)

	rows = cell_label.shape[0]
	cols = cell_label.shape[1]

	# cell_seg = np.array([[0 for i in range(cols)] for j in range(rows)])
	guttae_seg = np.array([[0 for i in range(cols)] for j in range(rows)])

	# plt.imshow(guttae_label==1, cmap='gray')
	# plt.axis("off")
	# plt.show()
	
	for i in range(len(np.unique(guttae_label))):
	    
	    guttae_ii = np.nonzero(guttae_label == i)
	    ii = [guttae_ii[0][0], guttae_ii[1][0]]
	    
	    if guttae[ii[0]][ii[1]] and seg[ii[0]][ii[1]]:
	        guttae_seg[seg_label == seg_label[ii[0]][ii[1]]] = 1
	#     else:
	#         cell_seg[seg_label == seg_label[ii[0]][ii[1]]] = 1

	cell_seg = seg - guttae_seg

	# fig, axs = plt.subplots(1, 2, figsize=(10, 10))

	# axs[0].imshow(cell_seg, cmap='gray')
	# axs[0].axis("off")

	# axs[1].imshow(guttae_seg, cmap='gray')
	# axs[1].axis("off")

	# plt.show()

	return cell_seg, guttae_seg, seg



def patch_predict(image, cellbound, guttaebound, model, patch_binarize=False, patch_size=(96,96), step=96):
    
    # Only usable for images of size multiple of 96. Otherwise, try changing the stepsize
    #print("salu2")
    patches = patchify(image, patch_size=patch_size, step=step)
    
    predicted_patches = []
    predicted_cell_patches = []
    predicted_guttae_patches = []
    
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            
            single_patch = patches[i,j,:,:]
            single_patch = np.expand_dims(single_patch, axis = 0)
            
            single_patch_prediction = np.squeeze(model.predict(single_patch, batch_size=1)[0,...])#.astype(np.uint8)
            
            # Binarize for cells and guttae
            if patch_binarize:
                
                cell_patch = (single_patch_prediction > cellbound)#.astype(np.uint8)
                guttae_patch = (single_patch_prediction < guttaebound)#.astype(np.uint8)
                
                single_patch_prediction = cell_patch + guttae_patch
                
                ##
                predicted_cell_patches.append(cell_patch)
                predicted_guttae_patches.append(guttae_patch)
                
            predicted_patches.append(single_patch_prediction)
            
    if patch_binarize:
        
        predicted_cell_patches = np.array(predicted_cell_patches)
        predicted_guttae_patches = np.array(predicted_guttae_patches)
        
        predicted_cell_patches_reshaped = np.reshape(predicted_cell_patches, patches.shape)
        predicted_guttae_patches_reshaped = np.reshape(predicted_guttae_patches, patches.shape)
        
        reconstructed_cell_image = unpatchify(predicted_cell_patches_reshaped, image.shape)
        reconstructed_guttae_image = unpatchify(predicted_guttae_patches_reshaped, image.shape)
        
        
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, patches.shape)
    reconstructed_image = unpatchify(predicted_patches_reshaped, image.shape)
#     plt.imshow(reconstructed_image)
            
    if patch_binarize:
        return reconstructed_cell_image, reconstructed_guttae_image, reconstructed_image
    else:
        return reconstructed_image



def watershed(image, cell, guttae, th_cell=0.15, th_guttae=0.25):

	sure_bg = np.array([[0 for i in range(cell.shape[1])] for j in range(cell.shape[0])])

	fg = cell + guttae

	cell_dist = cv2.distanceTransform(cell.astype(np.uint8), cv2.DIST_L2, 3)
	guttae_dist = cv2.distanceTransform(guttae.astype(np.uint8), cv2.DIST_L2, 3)

	ret2, sure_cell = cv2.threshold(cell_dist, th_cell*cell_dist.max(), 255, 0)
	ret3, sure_guttae = cv2.threshold(guttae_dist, th_guttae*guttae_dist.max(), 255, 0)

	sure_fg = np.uint8(sure_cell + sure_guttae)

	unknown = 1 - sure_bg/255 - sure_fg/255

	ret4, markers = cv2.connectedComponents(sure_fg)

	markers = markers + 10
	markers[unknown==1] = 0

	img = np.expand_dims(image, axis=-1)
	img = np.concatenate((img, img, img), axis=-1)
	markers = cv2.watershed(np.array(img).astype(np.uint8), markers)

	return sure_cell, sure_guttae, markers

