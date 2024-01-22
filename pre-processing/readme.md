# Pre-Processing

There are two main different processings detailled here:
- Image Registration (Applied only on the Cardiac Fat Dataset)
- General Data Organization to convert Dicom files to .tif files.

## Image Registration

## Methods
Given the mismatch between the labels and original DICOMs due to the manual cropping and centering of CT slices for labelling, an image registration methodology was used to align the labels to the DICOM images. In this way, full DICOM images can be used to train a deep learning segmentation model without the need for centering and cropping on external datasets.

First, the DICOM images were converted to the same range as labeled images of [-200, -30] HU to facilitate the recognition of key points between the two images. Image registration was then performed between pairs of DICOM and labeled CTs using the ORB (Oriented FAST and Rotated BRIEF) algorithm. The process consists first of finding the key points using an algorithm called FAST, which mainly uses an intensity threshold between the central pixel and a circular ring around the center. Then, ORB uses BRIEF (Binary Robust Independent Elementary Features) descriptors to describe each keypoint. BRIEF is a bit string description constructed from a set of binary intensity tests between $n$ pairs $(x,y)$ of pixels which are located inside the patch (a key point is in the center of the patch). For further details, the reader is referred to the original publications of the ORB algorithm \cite{rublee2011orb}. Next, a brute-force matcher was used to match the key points of the two images, selecting the best results while removing the noisy ones. The default parameters and the hamming distance were used for the ORB and the matcher, respectively. Finally, using the RANSAC algorithm it is possible to highlight the inliers (correct matches) and find the transformation matrix.

![registration_c](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/319cd54e-2e51-4369-a86a-f7572a3cefaf)

 **Figure 1.**          *DICOM at range (-1000,+1000) HU on the left, conversion to (-200,-30 HU) in the middle and the corresponding fat image(same proportions as the one containing the labels) on the right*


![match_points](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/5da8e401-5f6d-4adc-9b39-0a788d3aa31c)

**Figure 2.** *Example of the matching of some key points between the two images using BRFMatcher.*

It should be noted that the images are misaligned in exactly the same way within each patient, therefore only one manually chosen transformation matrix is applied per patient. With this, there will be 20 transformation matrices corresponding to the 20 patients in the database.

![registration](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/f15b0742-0e53-4c25-96f6-6c96166d6231)

### Results

**Figure 3.** *Two examples of the application of image registration. On the left both labels aren´t regist and they aren´t align with the DICOM image. On the right both labels are aligned with the DICOM image after performing registration.*

As you can see more clearly from Figure 3 before the application of image registration for this patient, the labels were completely out of alignment with the DICOM image. After the transformation matrix was applied to the same mask, the correct alignment of the mask with the DICOM image is verified as can be seen in the Figure 3. These results were observed for all patients in a qualitative way, so it was considered that this initial phase of the work was done, and the database organized for the following application. 

## General Data Pre-Processing 

The entire CT dataset underwent conversion from the DICOM format, which encompasses a broad range of values, to the 16-bit .tif format, facilitating Deep Learning training. Additionally, all data was arranged as outlined in the data folder. The code for this organization can be found in [data_organization.py](data_organization.py).
