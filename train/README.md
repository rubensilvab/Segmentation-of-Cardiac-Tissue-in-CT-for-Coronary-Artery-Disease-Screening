# Brief summary of Training Method

For the training, validation and testing of the model, each one of the Cardiac Fat and OSIC datasets were randomly divided as follows: 60\% of the CT scans for training, 20\% of the CT scans for validation, and the remaining 20\% for testing. A patient-wise division was applied to avoid bias given that different CT slices of the same patient are highly correlated.

The CHVNGE data was used for external testing of the automatic segmentation network and were thus not used for training.

Pericardial segmentation was then performed using a U-Net architecture. The U-Net is a popular framework for deep learning models, it often obtains excellent performance in image segmentation, especially in the area of medical image processing. This model was trained from scratch with a total number of 31,031,685 parameters.

Training parameters:
- Loss: Dice-Loss
- Optimizer: ADAM
- Early Stopping callback: stop training when the validation loss did not improve for 20 consecutive epochs.
- For training and validation, a custom image generator was utilized: employed an alternating approach for each consecutive step in the epoch, switching between images from the two datasets - avoids overfitting to the majority dataset
- Image size: 256x256
- Batch size = 12
- Learning rate of 1e-4

## Data Augmentations

### Traditional data augmentations
In our case the following transformations were employed in the training images: rotations within a range of 5 degrees, horizontal and vertical flip, horizontal and vertical shifts within a range of 10\% of the total length of the image, zoom within a range of 20\% (more or less the size of the image).

![augmentation](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/d1bd960c-3237-4438-91fc-303b90dcbc59)

### Artifitial Calcifications
The presence of calcifications, mainly on the CHVNGE dataset, negatively affected the segmentation performance.
Therefore, a innovative solution was to artificially create these calcifications and add them to the training data.

After analysing the pixel values of the calcifications it was evident that they represent high HU values, close to the limit of our window of HU (1000). Besides that, it was decided to replicate these calcifications with a Gaussian distribution. 
What is proposed on this work is to create these high intensity values with a 2D Gaussian shape and add to the CT slice.

![calcifications](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/29c33440-9542-4b8d-8e46-ea7ae4ab6fe7)



## Experiments

One of the main issues of the 2D model was often the incapacity to recognize the limits of the heart on the CT scan, leading to an over segmentation of the pericardium. Therefore two different preprocessing steps were proposed to help the model to focus only on the slices containing the pericardium.

### Slice Classification Network
The first strategy adopted was to classify each of the CT slices as containing or not the pericardium by using a simple 2D CNN.

![slc+2 5d_21](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/a9004eb0-ac61-4f19-9c22-79ddfaffb972)


After training the model, to limit the boundaries of the heart, the first and last slice containing the pericardium, were selected and all the slices in between were input to a pericardium segmentation model such as the 2D or 2.5D models proposed previously.
### 3D U-Net

The main objective of this 3D U-Net was not to provide an accurate 3D pericardial segmentation but rather to leverage the rich contextual information it offers to accurately identify the upper and lower boundaries of the pericardium. Similarly to the previous approach, the pericardium is then accurately segmented using the 2/2.5D U-Net.

![3d_2 5d](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/bef36081-51b2-4510-8fee-8b68a137021e)



