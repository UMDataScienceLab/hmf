# Implementation of the paper *Heterogeneous Matrix Factorization: When Features Differ by Datasets*

This repository contains the implementation of the paper *Heterogeneous Matrix Factorization: When Features Differ by Datasets*. 


# Files

solver.py implements the algorithm heterogeneous_matrix_factorization. main.py provides a few examples that call this function. For instance, running

```
python3 main.py --dataset=synthetic --logoutput=True
```
will apply hmf on the synthetic example in Section 6.1.

torchimgpro.py, emailprocess.py, stockprocess.py are used for loading and preprocessing the video frames, email communication networks, and stock prices.

# Data

The video frames are downloaded from [BMC dataset](http://backgroundmodelschallenge.eu/#learning)

To run the experiment on video segmentation, firstly download the video from the BMC dataset, next extract the frames in the video into jpg files and name these jpg files as 0.jpg, 1.jpg, ... Then change frame_folder in line 14 of torchimgpro.py to the folder that contains all jpg files. Finally, run
```
python3 main.py --dataset=video --logoutput=True
```


The email network data are downloaded from [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/email-Eu-core-temporal.html).

To run the experiment on temporal graph feature extraction, firstly download the email communication record from the SLNP dataset, next extract the txt file. Then change line 14 of emailprocess.py to the folder that contains txt file. Finally, run
```
python3 main.py --dataset=email --logoutput=True
```