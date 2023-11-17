# A Deep-Learning Approach to Marble-Burying Quantification: Image Segmentation for Marble and Bedding
  Workflow: load the marble/sawdust labels into the marble/sawdust folder in 'Labels'-> load the marble&sawdust labels into the 'Labels-double' folder-> Load the pre-processed inputs to the input folder, make sure each sets were renamed using continuous integers, like '0.tif', '1.tif'


This repository contains the following folders/files    
##  **Folders**     

  -FilesWithOriginalNaming: the folder with all used images for this project.  
  
  -Labels-double: with sub-folder @marbles and @sawdust, in each of the folders, the labels of marbles and sawdusts were listed using the same naming methods, corresponding with the sequential naming of images. The labels were all pre-processed and padded.   
  
  -Labels: The labels of marble and sawdust information in a single image, black area as sawdusts, and white areas within the black portion as marbles.  
  
  -inputs: Padded, renamed original color images.  
  
  -test:  A separated folder for small scale testing: for example, testing the prediciton of the model on one image.  
  
  -testresult-Yicheng'machine: contains a csv for loss and result data from one test on Yicheng's machine. The xlsx spreadsheet is the overall result generated.  
  
    
 ## **Files**
  -Convert_combine.m  //a matlab script to binarize the two seperate labels and merge them into one png file.  
      -when changing the k value, also update the variable k here.  
      
  -compare.m
  
  -main.py //the main program should be ran after all the original images and labels are in place. **uncomment the line 28 and line 88 for kfoldGenerator() if the kfolders were not already generated. (as a mass amount of data, Kfolders were not included in this repository)** 
