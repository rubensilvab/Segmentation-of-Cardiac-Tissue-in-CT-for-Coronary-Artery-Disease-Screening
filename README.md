# Segmentation of Cardiac Tissue in CT for Coronary Artery Disease Screening  
Please be aware that numerous variations and experiments were conducted during this study. However, only the automatic method that yielded the most favorable outcome has been included in this summary of the work. To access the entire study, please visit: https://hdl.handle.net/10216/153041.

## Motivation

Recent research indicates a connection between
epicardial adipose tissue (EAT) and Coronary Artery Disease
(CAD). EAT is a type of fat situated within the pericardium, a
thin membrane sac that covers the heart. Hence, its segmentation
and quantification could prove valuable for investigating its
potential as a CAD risk stratification tool. However, manually
segmenting these structures proves to be a demanding and time consuming
task, making it unsuitable for clinical settings. This
has driven the development of automated segmentation methods.
This study introduces an automated method for segmenting EAT
in CT scans.

## Methods

### Dataset

Three datasets were used in this study: Cardiac Fat,
OSIC and Centro Hospital de Vila Nova de Gaia e Espinho
(CHVNGE). The Cardiac Fat [^1] dataset includes 20 CT scans
with 878 slices belonging to 20 patients as DICOM images.
The original ground truth was obtained via manual segmentation
by a physician and a computer scientist who labeled the
EAT and pericardium. The OSIC [^2] dataset consists of 85 CT
scans with 12,133 slices whose scans were conducted using
six distinct scanners. The manual pericardial segmentation
were performed by an experienced radiologist. Finally, the
CHVNGE is a private dataset that includes 190 CT scans
with 8661 slices as DICOM images collected at the CHVNGE
in Vila Nova de Gaia, Portugal. The pericardial segmentation
was obtained via manual segmentation by a medicine student.
The model was exclusively trained using public datasets. The
Cardiac Fat and OSIC datasets were randomly divided with
60% of the CT scans for training, 20% for validation and the
remaining for testing.

### Pericardium segmentation

First, an automatic method was developed to accurately
segment the pericardium. Before training the network, the CT
slices were clipped to [-1000, 1000] HU and then normalized
to a range between 0 and 1. The input images were
resized to 256×256 and data augmentation techniques such
as rotations, zoom, flips and shifts were employed. Besides
that, calcifications were artificially generated using a Gaussian
distribution to make the model robust to the presence of
extensive calcifications and medical devices. A U-Net was then trained for pericardial segmentation [^3],
provided with three consecutive axial slices: the one to be segmented (k), as well as the previous (k - 1) and next (k +
1).
A post-processing technique was utilized in three key steps
to enhance the quality of the 3D image segmentation. Initially,
the largest connected component in the 3D space was retained,
discarding disconnected parts. To ensure continuity, a 3D
approach was implemented, incorporating pixels from adjacent
slices when present in both upper and lower slices. Lastly,
a 2D convex hull operation was applied to individual slices,
addressing holes and refining the segmentation’s appearance.
External validation was carried out on the 190 patients from
the CHVNGE dataset, employing the Dice Similarity Coefficient
(DSC), Hausdorff Distance (HD), and Mean Absolute
Distance (MAD) as evaluation metrics.

![2.5D Unet](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/66f5c67c-46ea-40c2-8981-d24181c74cd0.png)
                      **Figure 1.**          *Pericardial segmentation using the 2.5D U-Net*


### EAT segmentation

Once the pericardium was accurately segmented, the fat HU
range [-150,-50] was applied within the pericardium to isolate
the EAT. Figure 1 depicts the proposed approach.
All the external validation was conducted using the
CHVNGE dataset. Besides the initial 190 CT scans segmentations,
20 cases received secondary segmentation by both the
same student and a second specialist. This approach allows
for a comprehensive study of intra and interreader variability.
Evaluation of EAT segmentation performance was done
using the DSC, precision and recall. Subsequently, the quantification
of EAT volume was performed. The assessment
of agreement between the readers was conducted using the
Pearson Correlation Coefficient (PCC) and the bias.
![wflow](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/f353b635-2ae3-48c2-b149-0bf1fd8b8ab6)
 **Figure 2.**          *Workflow of the proposed method for automatic EAT segmentation*

## Results and discussion

The pericardium and EAT segmentation results are presented in Table 1 and Table 2, respectively.

<img width="550" alt="per_results" src="https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/ce992a97-e666-4d5a-a9aa-552fd68fb20b">

 **Table 1.**          *Pericardial segmentation results*

<img width="550" alt="eat_res" src="https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/fb716faf-3304-43bf-a607-12e6e3e3e06e">

**Table 2.**          *EAT segmentation results*

The pericardium segmentation achieved satisfatory results
for the CHVNGE dataset with DSC, HD, and MAD values
of 0.909 ± 0.024, 32.443 ± 13.729 mm, and 4.389 ± 1.291
mm, respectively.
Turning now to the EAT results, the quantification metrics
yielded a bias of 0.98 ± 15.351 cm3 and a PCC of 0.924
for the automatic method, demonstrating favorable agreement between manual quantification and the automated approach.
Regarding the segmentation metrics presented in Table I, they
reveal substantial intra and interreader variability, highlighting
the challenge of accurately delineating the pericardium and
EAT, even for specialists. Among the segmentation metrics, recall
showed the most similarity between the automatic method
and interreader performance. This finding implies that the
model’s capability to precisely identify EAT closely matches
that of an independent reader. Nevertheless, the lower precision
score, suggests occasional situations where the model
incorrectly classifies non-EAT fats as EAT. This primarily
happens with patients who exhibit anatomical variations, and
in the lower slices where other organs may be present.

![1 (9)](https://github.com/rubensilvab/Pericardial-Segmentation/assets/130314085/5897322a-e652-4bb1-9d8c-afba863d06e0)
**Figure 3.**          *Examples of EAT segmentation from the manual (top row) and automatic (bottom row) approach*

Figure 3 showcases three examples of EAT segmentation from different patients. In the first scenario, there is general agreement among segmentations, although a slightly larger EAT volume was identified by the human reader. The second instance illustrates the accuracy of the automatic model to deal with huge calcifications, presenting an almost perfect EAT segmentation. Additionally, the minimal impact of zoom and rotation on the model in these two cases underscores the efficacy of data augmentation methods. The third example exposes a deficiency in the automated approach, inaccurately segmenting EAT due to anatomical variations, being this a reason for the lower precision values. These variations are primarily observed in the lower slices where other organs might be present, and they occasionally appear in higher regions than usual due to certain medical conditions. This can lead the model to misidentify them as the pericardium. Therefore, this instance emphasizes the need for broader training data, encompassing diverse anatomical variations that conventional augmentation techniques cannot simulate.

## Conclusion

In conclusion, a successful model was developed for automated
EAT segmentation. The evaluation of this model’s
performance on 190 patients from the CHVNGE dataset
demonstrated satisfactory and promising results. However, it
is not yet suitable for clinical use. The use of artificial calcifications
and augmentation techniques during training proved
effective. Nevertheless, further training with more anatomical
variability data is necessary to enhance performance in patients
with diverse anatomies. Given the challenges of training a 3D
network, future work might involve incorporating additional
slices or different views on the 2.5D U-Net to help recognize
the pericardium in lower slices. Notably, this study’s primary
contribution lies in using publicly available data exclusively
for training the model and evaluation on an external private
dataset, making direct comparisons with existing literature
results challenging.

[^1]: O. Rodrigues, F. Morais, N. Morais, L. Conci, L. Neto, and A. Conci,
“A novel approach for the automated segmentation and volume quantification
of cardiac fats on computed tomography,” Computer Methods and
Programs in Biomedicine, vol. 123, pp. 109–128, Jan. 2016.
[^2]: Kónya et al. (2020), Lung segmentation dataset,
https://www.kaggle.com/sandorkonya/ct-lung-heart-trachea-segmentation,
(accessed Feb. 20, 2022).
[^3]: O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks
for biomedical image segmentation,” in International Conference on
Medical image computing and computer-assisted intervention. Springer,
2015, pp. 234–241.
