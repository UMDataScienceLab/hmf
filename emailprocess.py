import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchmetrics import ConfusionMatrix
from torch.autograd import Variable
import torchvision

def readdata():
    df=pd.read_csv('data/email/email-Eu-core-temporal.txt',header=None,sep=' ')
    return df

def df2graph(df):
    senders = df[0]
    #print(senders)
    receivers = df[1]
    ns = senders.max()+1
    nr = receivers.max()+1
    densemat = np.zeros((ns,nr))
    for i in range(len(df[0])):
        densemat[senders[i],receivers[i]] += 1
    return densemat
    

def df2data(df):
    maxtime = df[2].max()
    delta = 24*3600
    njour = maxtime // delta +1
    
    senders = df[0]
    receivers = df[1]
    ns = senders.max()+1
    nr = receivers.max()+1
    data = [np.zeros((ns,nr)) for j in range(njour)]
    
    for i in range(len(df[0])):        
        data[df[2][i]//delta][senders[i],receivers[i]] += 1
    data[df[2][i]//delta] -= np.diag(np.sum(data[df[2][i]//delta],axis=1))
    print("data are created from %s sources"%len(data))
    return data

def plotmat(mat,cm=plt.cm.Blues,suffix="",posinit=[],circle=True):
    #G = mat
    thres = 0.5
    mat[mat<=thres] = 0
    mat[mat>thres] = 1
    mat = mat-np.diag(np.diag(mat))
    G = nx.from_numpy_array(mat)
    if len(posinit) == 0 and (not circle):
        pos = nx.spring_layout(G, k=0.15, seed=4572321)
    else:
        pos = posinit

    colors = []
    for (u, v) in G.edges():
        colors.append(mat[u,v])
        #G[u][v]["color"] = mat[u,v]
    UG = G.to_undirected()
    #print(nx.number_connected_components(UG), "connected components")

    options = {
        "node_color": "black",
        "node_size": 1,
        #"edge_color": "gray",
        "edge_color":colors,
        "linewidths": 0,
        "with_labels" :False,
        "width": 1,
        "edge_cmap": cm,
        
    }
    if not circle:
        options["pos"]=pos
    plt.clf()

    figs,ax = plt.subplots(1,1, figsize=(8, 8),dpi=500) 
    print("drawing")
    if circle:
        nx.draw_circular(UG, ax=ax,**options)
    #nodes = list(UG.nodes())
    #edge_colors = [edgedata["color"] for _, _, edgedata in G.edges(data=True)]
    #nx.draw_networkx_edges(UG, width=2.0, edge_color=edge_colors)
    else:
        nx.draw_networkx(UG, ax=ax,**options)

    #plt.show()
    #print("saving")
    plt.savefig('emailgraph/email_%s.png'%suffix,bbox_inches="tight")
    return pos

    

def loademail(device):
    df = readdata()
    data = df2data(df)    
    Y=[torch.tensor(dfi,device=device).float() for dfi in data]
    return Y

def imshow(images, nrow=4):
    #img = img / 2 + 0.5     # unnormalize
    img = torchvision.utils.make_grid(images,nrow=nrow)
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(npimg)
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('emailcombines.png')

  

def plotall(Y, Ug,Vg,Ul,Vl):
    N = len(Vg)
    pos = []
    Ysum = sum([Y[i].cpu().detach().numpy() for i in range(N)])
    pos = plotmat(Ysum,suffix="aa.png",cm=plt.cm.RdPu,posinit=pos,circle=False)
            
    for i in range(N):
        print("saving %s ..."%i)
        with torch.no_grad():
            gc = Ug[i].lin_mat@Vg[i].lin_mat.T
            lc = Ul[i].lin_mat@Vl[i].lin_mat.T
            
            plotmat(Y[i].cpu().detach().numpy(),suffix="%s_full"%i,cm=plt.cm.RdPu,posinit=pos,circle=False)
            plotmat(gc.cpu().detach().numpy(),suffix="%s_shared"%i,posinit=pos,circle=False)
            plotmat(lc.cpu().detach().numpy(),suffix="%s_unique"%i,cm=plt.cm.Reds,posinit=pos,circle=False)
        if i>50:
            return    
    return
   
def test_err(Ytest,Ug,Vg,Ul,Vl,test2full,full2train,train2full,prevtrain,nexttrain):
    Nt = len(Ytest)
    res = 0
    for i in range(Nt):
        fid = test2full[i]
        
        previd = full2train[prevtrain[fid]]
        nextid = full2train[nexttrain[fid]]

        Ulpool = torch.cat((Ul[previd].lin_mat, Ul[nextid].lin_mat), dim=1)
        up,sp,vhp = torch.linalg.svd(Ulpool)
        r = len(Ul[previd].lin_mat[0])
        r = r //2
        Ulavg = up[:,:r]

        #Ulavg = (Ul[previd].lin_mat + Ul[nextid].lin_mat)/2
        Ugavg = Ug[previd].lin_mat
        Ucombine = torch.cat((Ugavg,Ulavg),dim=1)

        res += torch.norm(Ytest[i]-Ucombine@torch.linalg.pinv(Ucombine.T@Ucombine)@Ucombine.T@Ytest[i])

    return res/Nt
