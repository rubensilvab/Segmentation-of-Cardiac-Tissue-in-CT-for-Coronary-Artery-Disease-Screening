# Brief Description of the Content 

Please be aware that this work utilized three datasets; however, only one dataset is presented here, featuring only a subset of images. This serves as a mere illustration.
- Data
    - Name Dataset New (Example showcasing the distribution of data in .tif format)
         - 0 (Nr Fold for possibility of Cross Validation)
              -  Dicom
                  - Patient
                      - 0.tif 
                      - 1.tif
                      - ...NrSlice.tif
              -  Mask
                  - Patient
                      - 0.tif 
                      - 1.tif
                      - ...NrSlice.tif
              -  NameDataset.CSV - including all relevant information to facilitate the training process
                    - The fold designation for each patient and image
                    - Path of each image/slice
                    - If the image contains the pericardium or not
    - Name Dataset Raw (Example showcasing the distribution of data in the raw DICOM format .dcm )
         - Patient
              - 0.dcm
              - 1.dcm
              - ...NrSlice.dcm
         - GT/Patient
              - 0.bmp
              - 1.bmp
              - ...NrSlice.bmp
           
           
    
      
