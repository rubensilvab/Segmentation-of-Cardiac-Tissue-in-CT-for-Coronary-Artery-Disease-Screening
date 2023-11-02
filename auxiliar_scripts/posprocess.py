# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:12:50 2023

@author: RubenSilva
"""


# Load the 3D NRRD file
filename = 'X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/Peri_segm/NRRD/777072/777072_256_2.5UNet.nrrd'
# Load the 3D NRRD file
filename_manual = 'X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2.5D_Unet/Dice_loss/Hospital_tif/L0_W2000_calc_augm_tif/Peri_segm/NRRD/777072/777072_256_manual.nrrd'

#777072_256
def reshape_nrrd(nrrd):
  
  nrrd=np.array(nrrd)
  nrrd=np.transpose(nrrd,(2,1,0))
  #nrrd=nrrd[::-1]  
  return nrrd  

import cc3d
import numpy as np
import nrrd
from matplotlib import pyplot as plt
 #     # Read the data back from file
 #Predict
readdata, header = nrrd.read(filename)
#Manual
readdata2, header = nrrd.read(filename_manual)

labels_in=reshape_nrrd(readdata)
manual=reshape_nrrd(readdata2)

# Get a labeling of the k largest objects in the image.
# The output will be relabeled from 1 to N.
labels_out, N = cc3d.largest_k(
  labels_in, k=1, 
  connectivity=6, delta=0,
  return_N=True,
)

labels_out=labels_out.astype(np.uint8)

# for sli in range(labels_in.shape[0]):
    
#             fig=plt.figure(figsize=(10,10))
#             plt.subplot(1,3,1)
#             plt.imshow(manual[sli], cmap='gray')
#             plt.title('Manual')
#             plt.subplot(1,3,2)
#             plt.imshow(labels_in[sli], cmap='gray')
#             plt.title('2.5 UNet')
#             plt.subplot(1,3,3)
#             plt.imshow(labels_out[sli])
#             plt.title('Connected Components')
#             print(sli,np.unique(labels_out[sli]))
# print(N)

from scipy.spatial import ConvexHull

# Visualize the resulting convex hull
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = np.argwhere(labels_out == 1)

# # Extract the x, y, and z coordinates of the points
# x = idx[:, 1].reshape(-1,1)
# y = idx[:, 2].reshape(-1,1)
# z = idx[:, 0].reshape(-1,1)

# points= np.concatenate([x,y,z],axis=1)

# Compute the convex hull of the points
#hull = ConvexHull(points)

from scipy.spatial import ConvexHull

import cv2
import skimage.morphology, skimage.data
def fill_holes(labels_out):
    mask_fill=labels_out.copy()
    filled_img = np.zeros_like(mask_fill)
    for i in range(labels_out.shape[0]):
      # Find contours
        contours, _ = cv2.findContours(mask_fill[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on black image
        contour_img = np.zeros_like(mask_fill[i])
        cv2.drawContours(contour_img, contours, -1, 1, 1)
        
        # Fill enclosed regions
        
        for contour in contours:
            cv2.fillPoly(filled_img[i], [contour], 1)
          
    return filled_img.astype(np.uint8)


from PIL import Image, ImageDraw

def convex_hull_image(data):
    w,l=data.shape[0],data.shape[1]
    region = np.argwhere(data)
    try:   
        hull = ConvexHull(region)
        verts = [(region[v,0], region[v,1]) for v in hull.vertices]
        img = Image.new('L', data.shape, 0)
        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
        mask = np.array(img)
    except:    
        mask=np.zeros((w,l))
    return mask.T

def fill(labels_out):
    mask_fill=np.zeros(labels_out.shape)
    for i in range(labels_out.shape[0]):
        if( i==0 or i==labels_out.shape[0]-1):
            mask_fill[i,:,:]=labels_out[i,:,:]
        else:
            mask_fill[i,:,:]=np.logical_or(labels_out[i,:,:],np.logical_and(labels_out[i-1,:,:], labels_out[i+1,:,:]))
    return mask_fill.astype(np.uint8)

def convex_mask(labels_out):
    mask_convex=np.zeros(labels_out.shape)
    for i in range(labels_out.shape[0]):
            mask_convex[i,:,:]=convex_hull_image(labels_out[i,:,:])
    return mask_convex



mask_convex=convex_mask(labels_out)
mask_fill_2d=fill_holes(labels_out)

mask_fill_3d_fill=fill(labels_out) 
mask_fill_3d_c=convex_mask(mask_fill_3d_fill) 
# for sli in range(labels_in.shape[0]):
    
#             fig=plt.figure(figsize=(10,10))
#             plt.subplot(1,3,1)
#             plt.imshow(manual[sli], cmap='gray')
#             plt.title('Manual')
#             plt.subplot(1,3,2)
#             plt.imshow(labels_out[sli], cmap='gray')
#             plt.title('Connected Components')
#             # plt.subplot(1,3,3)
#             # plt.imshow(mask_fill[sli])
#             # plt.title('Connected Components + fill')
#             plt.subplot(1,3,3)
#             plt.imshow(mask_convex[sli])
#             plt.title('Connected Components+fill+ convex')
#             print(sli,np.unique(mask_fill[sli]))    







for sli in range(labels_in.shape[0]):
    
            fig=plt.figure(figsize=(20,10))
            plt.subplot(2,3,1)
            plt.imshow(manual[sli], cmap='gray')
            plt.title('Manual')
            plt.subplot(2,3,2)
            plt.imshow(labels_in[sli], cmap='gray')
            plt.title('2.5 UNet')
            plt.subplot(2,3,3)
            plt.imshow(labels_out[sli])
            plt.title('Connected Components')
            plt.subplot(2,3,4)
            plt.imshow(mask_convex[sli], cmap='gray')
            plt.title('Connected Components + convex')
            plt.subplot(2,3,5)
            plt.imshow(mask_fill_3d_fill[sli])
            plt.title('Connected Components + fill3d')
            plt.subplot(2,3,6)
            plt.imshow(mask_fill_3d_c[sli])
            plt.title('Connected Components+convex+fill3d')
            print(sli,np.unique(mask_fill_2d[sli]))    
            
            fig=plt.figure(figsize=(10,10))
            plt.imshow(mask_fill_3d_fill[sli])
            plt.title('Connected Components + fill3d')
            plt.imshow(mask_fill_3d_c[sli])
            plt.title('Connected Components+convex+fill3d')
import cv2



# # Create a binary mask with the same shape as the original data
# mask = np.zeros(labels_out.shape, dtype=np.uint8)

# # Loop over the slices and set the inside of the convex hull to ones
# for z in range(mask.shape[0]):
#     # Create a 2D array of x and y coordinates for the current slice
#     xx, yy = np.meshgrid(np.arange(mask.shape[2]), np.arange(mask.shape[1]))
#     # Check if each point in the current slice is inside the convex hull
#     idx = hull.find_simplex(np.column_stack((xx.ravel(), yy.ravel(), z*np.ones(xx.size)))) >= 0
#     # Set the inside of the convex hull to ones in the current slice
#     mask[z, yy.ravel()[idx], xx.ravel()[idx]] = 1

# # Extract slice by slice and save to a numpy array
# convex_slices = np.zeros((mask.shape[0], mask.shape[2], mask.shape[1]), dtype=np.uint8)
# for z in range(mask.shape[0]):
#     convex_slices[z] = mask[z].T
    
# # Plot the convex hull in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
# ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=hull.simplices, alpha=0.2)
# plt.show()



# # Plot the points in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, s=1)

# # Set the view of the plot
# ax.view_init(elev=10, azim=45)

# plt.show()

# import scipy.spatial as spatial

# # Compute the 3D convex hull of the binary volume
# points = np.argwhere(labels_out)
# hull = spatial.ConvexHull(points)

# # Create a new binary volume with the shape of the original volume
# convex_hull_image = np.zeros_like(labels_out)


# for h in np.unique(points[:,0]):
#     pointsh=points[np.where(points[:,0]==h)]
    
#     plt.plot(pointsh[:,1],pointsh[:,2],'o')



# for sli in range(labels_in.shape[0]):
    
#             fig=plt.figure(figsize=(10,10))
#             plt.subplot(1,2,1)
#             plt.imshow(labels_in[sli], cmap='gray')
#             plt.subplot(1,2,2)
#             plt.imshow(convex_hull_image[sli])
#             print(sli,np.unique(convex_hull_image[sli]))
# print(N)


# connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
# labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

# # If you need a particular dtype you can specify np.uint16, np.uint32, or np.uint64
# # You can go bigger, not smaller, than the default which is selected
# # to be the smallest that can be safely used. This can save you the copy
# # operation needed by labels_out.astype(...).
# labels_out = cc3d.connected_components(labels_in, out_dtype=np.uint64)

# # If you're working with continuously valued images like microscopy
# # images you can use cc3d to perform a very rough segmentation. 
# # If delta = 0, standard high speed processing. If delta > 0, then
# # neighbor voxel values <= delta are considered the same component.
# # The algorithm can be 2-10x slower though. Zero is considered
# # background and will not join to any other voxel.
# labels_out = cc3d.connected_components(labels_in, delta=10)

# # You can extract the number of labels (which is also the maximum 
# # label value) like so:
# labels_out, N = cc3d.connected_components(labels_in, return_N=True) # free
# # -- OR -- 
# labels_out = cc3d.connected_components(labels_in) 
# N = np.max(labels_out) # costs a full read

# # You can extract individual components using numpy operators
# # This approach is slow, but makes a mutable copy.
# for segid in range(1, N+1):
#   extracted_image = labels_out * (labels_out == segid)
#   process(extracted_image) # stand in for whatever you'd like to do

# # If a read-only image is ok, this approach is MUCH faster
# # if the image has many contiguous regions. A random image 
# # can be slower. binary=True yields binary images instead
# # of numbered images.
# for label, image in cc3d.each(labels_out, binary=False, in_place=True):
#   process(image) # stand in for whatever you'd like to do

# # Image statistics like voxel counts, bounding boxes, and centroids.
# stats = cc3d.statistics(labels_out)

# # Remove dust from the input image. Removes objects with
# # fewer than `threshold` voxels.
# labels_out = cc3d.dust(
#   labels_in, threshold=100, 
#   connectivity=26, in_place=False
# )

# # Get a labeling of the k largest objects in the image.
# # The output will be relabeled from 1 to N.
# labels_out, N = cc3d.largest_k(
#   labels_in, k=10, 
#   connectivity=26, delta=0,
#   return_N=True,
# )
# labels_in *= (labels_out > 0) # to get original labels

# # Compute the contact surface area between all labels.
# # Only face contacts are counted as edges and corners
# # have zero area. To get a simple count of all contacting
# # voxels, set `surface_area=False`. 
# # { (1,2): 16 } aka { (label_1, label_2): contact surface area }
# surface_per_contact = cc3d.contacts(
#   labels_out, connectivity=connectivity,
#   surface_area=True, anisotropy=(4,4,40)
# )
# # same as set(surface_per_contact.keys())
# edges = cc3d.region_graph(labels_out, connectivity=connectivity)

# # You can also generate a voxel connectivty graph that encodes
# # which directions are passable from a given voxel as a bitfield.
# # This could also be seen as a method of eroding voxels fractionally
# # based on their label adjacencies.
# # See help(cc3d.voxel_connectivity_graph) for details.
# graph = cc3d.voxel_connectivity_graph(labels, connectivity=connectivity)



# Save the new binary volume as a NRRD file
new_filename = 'X:/Ruben/TESE/New_training_Unet/Models_only_pub_dset/Mix_datasets/All_data/predict/2D_Unet/Dice_loss/Hospital_tif/L0_W2000_tif_calc_augm/new_with_resize/NRRD/282459/teste_posprocess.nrrd'
#nrrd.write(new_filename, reshape_nrrd(mask_fill_3d_c),header=header)

