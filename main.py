import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from solver import *

class Experiment():
 
    def synthetic(inputargs):
        # The experiment in section 6.1
        args = {
            "r1":3,
            "r2":3,
            "optim":"SGD",
            "lr":0.1,
            "epochs":500,
            "beta":1e-1,
            "seed":100,
            "verbose":2,
            "wd":0,
            #"epsilon":1e-8,
        }
        print(args)
        d = 120
        ndt = 100
        N = 100
        #Ugt = torch.randn(d,args["r1"])/np.sqrt(d)
        device = "cuda"

        Ugt = torch.zeros(d,args["r1"]).to(device)
        for i in range(args["r1"]):
            Ugt[i,i]=1#0

        Q, R = torch.linalg.qr(torch.randn(d,d))    
        Q = Q.to(device)
        Ult = torch.zeros(N, d, args["r2"]).to(device)
        ter_id = 119
        ini_id = 5
        for clid in range(N):
            for cpid in range(args["r2"]):
                Ult[clid, ((cpid + clid) % (ter_id - ini_id + 1) + ini_id), cpid] = 1
        with torch.no_grad():
            Ugt = Q@Ugt
            for i in range(N):
                Ult[i,:,:] = Q@Ult[i,:,:]
        import torch.nn.functional as F
        pavg = sum(Ult[i,:,:]@Ult[i,:,:].T for i in range(N))/N
        up,sp,vhp = torch.linalg.svd(pavg)
        #print(sp)
        print("theta is %.6f"%(1-sp[0]))
        
        
        Y = []
        ulst = []
        vgst = []
        vlst = []
        for i in range(N):
            Vgt = torch.randn(ndt,args["r1"]).to(device)#*100
            Qvg,R = torch.linalg.qr(Vgt)
            Vgt = Qvg
            Ult[i,:,:] -= Ugt @ torch.inverse(Ugt.T@Ugt) @ Ugt.T @ Ult[i,:,:]
            ulst.append(Ult[i,:,:])
            Vlt = torch.randn(ndt,args["r2"]).to(device)#*100
            Vlt -= Vgt @ torch.inverse(Vgt.T@Vgt) @ Vgt.T@Vlt
            Qvl,R = torch.linalg.qr(Vlt)
            Vlt = Qvl
            
            ygpart = Ugt@Vgt.T           

            yipart = Ult[i,:,:]@Vlt.T           
            
            Y.append(ygpart+yipart)
            vgst.append(Vgt)
            vlst.append(Vlt)
        

        args["global_subspace_err_metric"]= lambda x: subspace_error(Ugt,x)
        args["local_subspace_err_metric"] = lambda x: sum([subspace_error(ulst[i],x[i]) for i in range(N)])/N
     
        args["global_recovery_error"] = lambda x: g_recovery([Ugt,vgst,ulst,vlst],x,'f')
        args["local_recovery_error"] = lambda x: l_recovery([Ugt,vgst,ulst,vlst],x,'f')

        Ug,Vg,Ul,Vl = heterogeneous_matrix_factorization(Y,args,verbose=20)
   
    def video(inputargs):
        args = {
            "r1":20,
            "r2":100, # a little overparametrization on r2 is helpful
            "optim":"SGD",
            "lr":0.000005,
            "epochs":5000,
            "seed":100,
            "beta":1e-2,
            "wd":0,           
            "reshuffle":1,
            "verbose":1,
            "device":"cuda"
        }
        print(args)
        import torchimgpro 
        Y = torchimgpro.load_video_data(args)
        
        if args["reshuffle"]:
            args["n1"] = 7
            args["n2"] = 7
            n1 = args["n1"]
            n2 = args["n2"]
            (d1,d2) = Y[0].shape
            args["k1"] = d1 // n1
            args["k2"] = d2 // n2 
            Y = [torchimgpro.reshuffle(yi, n1, n2) for yi in Y]
        N = len(Y)
        print(Y[0].shape)
        maxelement = torch.max(torch.cat(Y).abs()).item()
        minelement = torch.min(torch.cat(Y).abs()).item()
        norms = []
        transmat = []
       
        print(Y[0])
        Ug,Vg,Ul,Vl = heterogeneous_matrix_factorization(Y,args)

        reconstruct_bg = [(Ug[i].lin_mat@Vg[i].lin_mat.T) for i in range(N)]
        reconstruct_cat = [(Ul[i].lin_mat@Vl[i].lin_mat.T) for i in range(N)]
        

        folder = r'processedframes/'
        with torch.no_grad():
            for i in range(N):
                print("saving figure [%s/%s]"%(i,N))

                original = Y[i].detach().cpu().numpy()
                torchimgpro.show_save(original,folder+'original_%s.png'%i,maxelement,minelement,args=args)
             
                reconstruct_bg[i] = reconstruct_bg[i].detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_bg[i],folder+'recons_bg_%s.png'%i,args=args)
               
                reconstruct_cat[i] = reconstruct_cat[i].detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_cat[i],folder+'recons_car_%s.png'%i,args=args)

 
    def email(inputargs):
        args = {
            "r1":50,
            "r2":10,
            "optim":"SGD",
            "lr":0.005,
            "beta":0.01,
            "epochs":1000,
            "seed":100,
            "verbose":2,
            "wd":0,
        }
        import emailprocess as ep

        Y = ep.loademail("cuda")   
        # filter out thoses dates with infrequent email communications
        Y = [Y[i] for i in range(len(Y)) if torch.sum(torch.abs(Y[i]))>40] 
        print("%s samples"%len(Y))
        N = len(Y)

        # train test split
        ptrain = 0.9
        Ytrain = []
        Ytest = []
        train2full = dict()

        test2full = dict()
        full2train = dict()

        ass = np.random.choice(2, N, p=[ptrain,1-ptrain])
        ass[0] = 0
        ass[N-1] = 0
        prevtrain = dict()
        nexttrain = dict()
        pt = 0
        for i in range(N):
            if ass[i] == 0:
                train2full[len(Ytrain)] = i
                full2train[i] = len(Ytrain)
                Ytrain.append(Y[i])    
                pt = i            
            else:
                prevtrain[i] = pt
                test2full[len(Ytest)] = i
                Ytest.append(Y[i])
        nt = N-1
        for i in range(N):
            if ass[N-1-i] == 0: 
                nt = N-1-i           
            else:
                nexttrain[N-1-i] = nt
                
        
        print("%s trainning samples and %s testing samples"%(len(Ytrain),len(Ytest)))

        
        print(args)
        Ug,Vg,Ul,Vl= heterogeneous_matrix_factorization(Ytrain,args)
        terr = ep.test_err(Ytest,Ug,Vg,Ul,Vl,test2full,full2train,train2full,prevtrain,nexttrain)
        print('hmf: test error %.4f'%terr)


        sumtrain = sum(Ytrain)
        upool,spool,vhpool = torch.svd(sumtrain)
        upool = upool[:,:(args['r1']+args['r2'])]
        err = sum([torch.norm(yi - upool@upool.T@yi) for yi in Ytest])/len(Ytest)
        print('pooled mf: test error %.4f'%err)
        ep.plotall(Y,Ug,Vg,Ul,Vl)
        return
      
    def stock(inputargs):
        args = {
            "r1":10,
            "r2":10,
            "optim":"SGD",
            "lr":0.001,
            "beta":0.01,
            "epochs":10000,
            "seed":100,
            "verbose":2,
            "wd":0,
           
        }
        import stockprocess as sp

        Y,rawdf = sp.loadstocks("cuda")   
        print("%s samples"%len(Y))
        print(Y[0].shape)
        N = len(Y)
        Ug,Vg,Ul,Vl = heterogeneous_matrix_factorization(Y,args)       
    
        lfac = []
        
        for i in range(N):
            si = torch.sum(torch.abs(Ul[i].lin_mat@Vl[i].lin_mat.T).cpu(),dim=0)
            lfac.append(si)
        sp.plotall(rawdf,lfac)
        return 
     

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='heterogeneous matrix factorization')
    parser.add_argument('--dataset', type=str, default="synthetic")
    parser.add_argument('--algorithm', type=str, default="dgd")
    parser.add_argument('--logoutput', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--d', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--num_client', type=int, default=100)
    parser.add_argument('--r2', type=int, default=1)
    parser.add_argument('--r1', type=int, default=1)
    
    parser.add_argument('--num_dp_per_client', type=int, default=1000)
    parser.add_argument('--folderprefix', type=str, default='')

    args = parser.parse_args()
    args = vars(args)
    if args['logoutput']:
        import os
        from misc import Tee
        import time
        import sys
        output_dir = args['folderprefix']+'outputs/{}_'.format(args['dataset'])
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt')) 

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment = getattr(Experiment, args['dataset'])
    experiment(args)
  
   
