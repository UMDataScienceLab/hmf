import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy
import os
import torch
import torch.nn as nn


def load_video_data(args={}):    
    names = [str(i)+".jpg" for i in range(62)]
    res = []
    for name in names:
        frame_folder = r'data/frames/'
        if not os.path.isfile(frame_folder+name):
            continue
        img = Image.open(frame_folder+name)
    
        img = np.array(img)
        print(img.shape)
        cat = img
        if len(cat.shape) > 2:
            cat = np.mean(img, axis=2)
        ct = cat#.T
        
        res.append(torch.tensor(ct).float().to(args["device"]))

    print('images loaded')
    return res


def reshuffle(Y,n1,n2):
    (d1,d2) = Y.shape
    k1 = d1 // n1
    k2 = d2 // n2
    truncate = Y[:k1*n1,:k2*n2]
    ufd = nn.Unfold(kernel_size=(k1,k2),dilation=(n1,n2))
    return ufd(truncate.unsqueeze(0).unsqueeze(0))[0]#.T
   
def shuffleback(Yshuffled, n1, n2, k1, k2):
    Ys = Yshuffled#.T
    fd = nn.Fold(output_size=(n1*k1,n2*k2),kernel_size=(k1,k2),dilation=(n1,n2))
    return fd(torch.tensor(Ys).unsqueeze(0)).numpy()[0][0]
  

def show_save(pic,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    picture = pic.copy()
    if "reshuffle" in args and args["reshuffle"]:
        picture = shuffleback(picture, args["n1"], args["n2"], args["k1"], args["k2"])
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        plt.imshow(picture, cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.show()
    else:
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        plt.imshow(picture, cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.show()

