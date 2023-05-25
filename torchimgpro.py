import matplotlib.pyplot as plt
from PIL import Image
#img = Image.open('data/stinkbug.png')
import numpy as np
import copy
import os
import cv2
import torch
from scipy import fftpack, ndimage
import torch.nn as nn
import pickle

def img_add(front, back, stx, sty):
    w, h = front.shape
    res = copy.deepcopy(back)
    res[stx:stx+w, sty:sty+h] = res[stx:stx+w, sty:sty+h]+front
    return res


def gen_img_data(args={}):
    return load_girl_data(args)
    # return load_car_data()
    # return load_cat_data()


def save_image(image,addr,num):
    cv2.imwrite(addr+str(num)+".jpg", image)

def cv2_save_img(image, addr):
    cv2.imwrite(addr, image)

def load_girl_data(args={}):
    #names = ["image00"+str(i).zfill(3)+".jpg" for i in range(10,202)]
    #names = ["I_MC_02-"+str(i).zfill(3)+".bmp" for i in range(157,240, 2)]
    #names = [str(i).zfill(4)+".jpg" for i in range(208,245)]
    #names = ["I_SI_01-"+str(i).zfill(3)+".bmp" for i in range(78, 294)]
    #names = ["I_SM_01-"+str(i)+".bmp" for i in range(56, 295)]
    #names = ["in"+str(i).zfill(6)+".jpg" for i in range(0,801)]
    #names = ["in"+str(i).zfill(6)+".jpg" for i in range(1259,1980)]

    #names = ["sample"+str(i)+'.png' for i in [1,2,3,8]]
    names = [str(i)+".jpg" for i in range(62)]
    #names = [str(i)+".jpg" for i in range(4)]

    #names = ["img"+str(i)+'.png' for i in [9,10,11,12,13,14,15,16]]
    #names = ["img"+str(i)+'.png' for i in range(17,21)]
    #names = ["AMBench_625_Build1_Layer361_frame"+str(i)+'.jpg' for i in range(624,634)]
    res = []
    for name in names:
        #img = Image.open(r'frames/fading/'+name)
        #img = Image.open(r'frames/shaking/'+name)
        #img = Image.open(r'frames/simplewalk/'+name)
        pca_folder = r'../../pca/working/perpca/frames/'
        #pca_folder = r'images_ds/'
        if not os.path.isfile(pca_folder+r'/car3/'+name):
            continue
        img = Image.open(pca_folder+r'/car3/'+name)
    
        img = np.array(img)
        print(img.shape)
        cat = img
        if len(cat.shape) > 2:
            cat = np.mean(img, axis=2)
        #cat = cat[:,:600]
        ct = cat#.T
        
        res.append(torch.tensor(ct).float().to(args["device"]))

        #print(cat.shape)
    #print(len(res))
    print('images loaded')
    return res

def load_thermal_data(resdict):
    from os import listdir
    from os.path import isfile, join
    import re    
    srcpath = r'images_ds/test1_thermal/'
    onlyfiles = [join(srcpath, f) for f in listdir(srcpath) if isfile(join(srcpath, f))]
    name_pattern = re.compile(".Build([0-9]+)_Layer([0-9]+)_([0-9]+).pkl")
    totnum = 0
    for fi in onlyfiles:
        s = name_pattern.search(fi)
        if s:
            bd = int(s.group(1))
            ly = int(s.group(2))
            fm = int(s.group(3))
            
            if not bd in resdict.keys():
                resdict[bd] = dict()
            if not ly in resdict[bd].keys():
                resdict[bd][ly] = dict()
            with open(fi, 'rb') as fin :
                img = pickle.load(fin)
            img = np.array(img)
            #print(img.shape)
            cat = img
            #if len(cat.shape) > 2:
            #    cat = np.mean(img, axis=2)
            ct = cat#.T
            resdict[bd][ly][fm] = torch.tensor(cat).float()
            totnum+=1
            #print(bd,ly,fm)
            #print(bd, ly, len(resdict[bd][ly].keys()))
    print("%s files loaded from %s"%(totnum, srcpath))
    for bd in resdict:
        for ly in resdict[bd]:
            print(bd, ly, len(resdict[bd][ly].keys()))
    return resdict
    
def load_thermal_data_old(resdict):
    from os import listdir
    from os.path import isfile, join
    import re    
    srcpath = r'../../pca/working/perpca/frames/thermal/NIST Build2 Layers251-280/'
    onlyfiles = [join(srcpath, f) for f in listdir(srcpath) if isfile(join(srcpath, f))]
    name_pattern = re.compile(".AMB2018_625_Build([0-9]+)_Layer([0-9]+)_frame([0-9]+).jpg")
    totnum = 0
    for fi in onlyfiles:
        s = name_pattern.search(fi)
        if s:
            bd = int(s.group(1))
            ly = int(s.group(2))
            fm = int(s.group(3))
            if not bd in resdict.keys():
                resdict[bd] = dict()
            if not ly in resdict[bd].keys():
                resdict[bd][ly] = dict()
            img = Image.open(fi)    
            img = np.array(img)
            #print(img.shape)
            cat = img
            if len(cat.shape) > 2:
                cat = np.mean(img, axis=2)
            ct = cat#.T
            resdict[bd][ly][fm] = torch.tensor(cat).float()
            totnum+=1
            #print(bd,ly,fm)
            #print(bd, ly, len(resdict[bd][ly].keys()))
    print("%s files loaded from %s"%(totnum, srcpath))
    for bd in resdict:
        for ly in resdict[bd]:
            print(bd, ly, len(resdict[bd][ly].keys()))
    return resdict

def imgsshow(compose):
    for c in compose:
        plt.imshow(c)
        plt.axis('off')
        plt.show()


def threshold(file):
    img = Image.open(file)
    
    img = np.array(img)
    img = np.mean(img, axis=2)
    #print(np.max(img))
    #print(np.min(img))
    print(img.shape)
    fmask = img>=130
    #print(fmask[150]*255)
    print(fmask*255)
    print(fmask.mean())
    plt.imshow(img[:,0:600],cmap='gray')
    plt.savefig('trexample3.png')


def gen_ellipses():
    image = cv2.imread("bg2.jpg")
    image = cv2.resize(image, (800,800), interpolation = cv2.INTER_AREA)
    N = 100
    for i in range(N):
        overlay = image.copy()
        ri = torch.randint(800, (2,))
        center_coordinates = (ri[0].item(), ri[1].item())
        axesLength = (400, 200)  
        angle = torch.rand(1).item()*360
        startAngle = 0  
        endAngle = 360   
        # Blue color in BGR
        color = (255, 255, 0)    
        # Line thickness of -1 px
        thickness = -1    
        # Using cv2.ellipse() method
        # Draw a ellipse with blue line borders of thickness of -1 px
        overlay = cv2.ellipse(overlay,center_coordinates, axesLength, angle,
                                startAngle, endAngle, color, thickness)
        # transparency
        alpha = 0.9
        #new image
        image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        cv2.imwrite("images_ds/ellipses/img_%s.jpg"%(i+1),image_new)

def position2momentum(Y):
    res = dict()
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
    for ki in alliters:
        timage = Y[ki]
        image = timage.numpy()
        fft2 = fftpack.fft2(image)
        fft2r = np.real(fft2)
        fft2img = np.imag(fft2)
        cbd = np.concatenate((fft2r,fft2img), axis=0)
        res[ki]=torch.tensor(cbd)
    return res

def r2handdictionary(Y, dictmult=10, sigma=0.1):
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
    # generate dictionaries
    gap = max(1,len(alliters)//dictmult)
    dictionary = []
    for i,ki in enumerate(alliters):
        if i%gap == 0:
            dictionary.append(Y[ki])
    dictionary = torch.cat(dictionary,dim=1)
    largeidx = torch.where(torch.norm(dictionary, dim=0) > 1e-4)[0]
    dictionary = dictionary[:,largeidx]
    tmax = torch.max(dictionary)
    dictionary = torch.abs(torch.randn(len(Y[ki]),10000))*tmax/3
    print("Dictionary built with dimension %s x %s"%(dictionary.shape[0],dictionary.shape[1]))
    with torch.no_grad():
        res = dict()
        for ki in alliters:
            timage = Y[ki]            
            dist = torch.cdist(dictionary.T, timage.T)
            res[ki]= torch.exp(-0.5*dist**2/sigma**2)
    #print(dictionary[:,0])
    #print('dictionary printed')
    #assert False
    return res, dictionary

def h2r(h, dictionary, sigma=0.1, eps=1e-3):
    (d,ndic) = dictionary.shape
    (ndic,nsample) = h.shape
    with torch.no_grad():
        z = torch.randn(d,nsample)*1e-2
        beta = 0.5
        for i in range(200):
            zdist = torch.cdist(dictionary.T,z.T) #  ndic x nsample 
            coeff = torch.exp(-0.5*zdist**2/sigma**2) * h #  ndic x nsample 
            coeff = (coeff/(1e-5+torch.sum(coeff, dim=0)))
            znew = dictionary@coeff # d x nsample
            if torch.norm(znew-z)<1e-5:
                break
            #print(i,torch.norm(z),torch.norm(znew-z)) 
            # use exponential averaging to stablize the iterate
            z = (1-beta)*z + beta*znew
            #z += znew       
            
    return znew

def hdict2r(Y, dictionary, sigma=0.1):
    if isinstance(Y, list):
        N = len(Y)
        alliters = list(range(N))
    else:
        alliters = Y.keys()
    res = dict()
    with torch.no_grad():
        res = dict()
        for ki in alliters:                      
            res[ki]=h2r(Y[ki], dictionary, sigma,eps=1e-3)
    return res

def reshuffle(Y,n1,n2):
    (d1,d2) = Y.shape
    k1 = d1 // n1
    k2 = d2 // n2
    truncate = Y[:k1*n1,:k2*n2]
    ufd = nn.Unfold(kernel_size=(k1,k2),dilation=(n1,n2))
    return ufd(truncate.unsqueeze(0).unsqueeze(0))[0]#.T
    mid1 = truncate.unfold(0,n1,n1)
    z = torch.zeros(k1*k2,n1*n2)
    print("transforming...")
    for x1 in range(n1):
        for x2 in range(n2):
            for j1 in range(k1):
                for j2 in range(k2):                    
                    z[j1*k2+j2,x1*n2+x2] += truncate[j1*n1+x1,j2*n2+x2]
    return z#.T

def shuffleback(Yshuffled, n1, n2, k1, k2):
    Ys = Yshuffled#.T
    fd = nn.Fold(output_size=(n1*k1,n2*k2),kernel_size=(k1,k2),dilation=(n1,n2))
    return fd(torch.tensor(Ys).unsqueeze(0)).numpy()[0][0]
    z = np.zeros((k1*n1,k2*n2))
    for x1 in range(n1):
        for x2 in range(n2):
            for j1 in range(k1):
                for j2 in range(k2):
                    z[j1*n1+x1,j2*n2+x2] += Ys[j1*k2+j2,x1*n2+x2]
    return z

    
def reconstruct_picture(pic,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    picture = pic.copy()    
    if "momentum" in args and args["momentum"]:
        (n1,n2) = pic.shape
        n1 = n1//2
        fftrecons = picture[:n1,:]+np.array([1j])*picture[n1:,:]
        fft3 = fftpack.ifft2(fftrecons)
        return abs(fft3)
    elif "reshuffle" in args and args["reshuffle"]:
        if "kernel" in args and args["kernel"]:
            #print(picture.shape)
            #print(torch.tensor(picture).shape)
            #print(args['kerneldict'].shape)
            picture = h2r(torch.tensor(picture), args['kerneldict'], args['sigma']).numpy()
        picture = shuffleback(picture, args["n1"], args["n2"], args["k1"], args["k2"])
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        return picture
    else:
        picture[picture>cutoff_up] = cutoff_up
        picture[picture<cutoff_low] = cutoff_low
        return picture


def only_show_save(picture,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    plt.imshow(picture, cmap='gray')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.show()


def show_save(pic,name,cutoff_up=1e6,cutoff_low=-1e6,args={}):
    picture = pic.copy()
    
    if "momentum" in args and args["momentum"]:
        (n1,n2) = pic.shape
        n1 = n1//2
        fftrecons = picture[:n1,:]+np.array([1j])*picture[n1:,:]
        fft3 = fftpack.ifft2(fftrecons)
        plt.imshow(abs(fft3), cmap='gray')
        plt.axis('off')
        plt.savefig(name, bbox_inches='tight')
        plt.show()
    elif "reshuffle" in args and args["reshuffle"]:
        if "kernel" in args and args["kernel"]:
            #print(picture.shape)
            #print(torch.tensor(picture).shape)
            #print(args['kerneldict'].shape)
            picture = h2r(torch.tensor(picture), args['kerneldict'], args['sigma']).numpy()
       
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

def show_fft():
    import matplotlib.pyplot as plt
    Y = load_girl_data()
    image = Y[0].numpy()
    fft2 = fftpack.fft2(image)
    print(fft2)
    fft2r = np.real(fft2)
    (n1,n2)=fft2r.shape
    fft2img = np.imag(fft2)
    cbd = np.concatenate((fft2r,fft2img))
    u,s,vt = np.linalg.svd(cbd,full_matrices=False)
    print(u.shape,s.shape,vt.shape)
    s[10:]*=0
    smat = np.diag(s)
    recons = u@smat@vt
    fftrecons = recons[:n1,:]+np.array([1j])*recons[n1:,:]


    plt.imshow(np.log10(abs(fftrecons)))
    plt.savefig('fft.png')
    fft3 = fftpack.ifft2(fftrecons)
    plt.imshow(abs(fft3))
    plt.savefig('fft3.png')

    u,s,vt = np.linalg.svd(image,full_matrices=False)
    #print(u.shape,s.shape,vt.shape)
    s[10:]*=0
    smat = np.diag(s)
    recons = u@smat@vt
    plt.imshow(recons)
    plt.savefig('fft4.png')

    



    '''
    image2 = Y[1].numpy()
    fft22 = fftpack.fft2(image2)

    plt.imshow(np.log10(abs(fft22)))
    plt.savefig('fft22.png')
    '''
    

if __name__ == "__main__":
    #imgsshow(gen_img_data())
    #process_cat_data_xb()
    #process_car_data()
    #process_office_data()
    #threshold(r'processedframes/rpca_cat_6.png')
    #threshold(r'frames/office/0.jpg')
    #gen_ellipses()
    #u = np.random.randn(100,2)
    #v = np.random.randn(100,2)
    #cv2.imwrite("random.jpg",np.abs(u@v.T)*256)
    #show_fft()
    load_thermal_data(dict())
