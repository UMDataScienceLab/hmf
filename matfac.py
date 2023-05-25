#basic libary
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from solver import *
#from plotting import *

class Experiment():


    def toy_nonoise(inputargs):
        args = {
            "ngc":1,
            "nlc":1,
            "optim":"SGD",
            "lr":0.1,
            "epochs":10000,
            "outer_epochs":500,
            "seed":100,
            "lbd_s":100000,
            "lbd_s_outer":10,
            "beta":0.01,
            "verbose":2,
            "rho":0.99,
            
            "wd":0,
            #"tensorboard":0,
            "normalize_inner_loop":0,
            'num_dp_per_client':10,
            'toytheta':60,

        }
        args['sparsity'] = inputargs['sparsity']
        args['toytheta'] = inputargs['toytheta']

        device = "cuda"
        d = 3
        Q, R = torch.linalg.qr(torch.randn(d,d))    
        Q = Q.to(device)
        gcs = torch.tensor([[0,0,1.]],device = device).float()
        theta = args['toytheta']/180*np.pi
        lcs = torch.tensor([[[np.cos(theta/2),np.sin(theta/2),0]],[[np.cos(theta/2),-np.sin(theta/2),0]]],device=device).float()
        
        pavg = (lcs[0].T@lcs[0]+lcs[1].T@lcs[1])/2
        up,sp,vhp = torch.linalg.svd(pavg)
        print("theta = %.8f"%(1-sp[0]))
        #return
        gsigma = 1
        lsigma = 1
        theta1 = torch.rand(args['num_dp_per_client'],device=device)*2*np.pi
        theta1 = theta1.view(1,len(theta1))
        # Y has dimension (n_client, num_dp, d)
        Y1 = gsigma*Q@gcs.T@torch.cos(theta1)+lsigma*Q@lcs[0].T@torch.sin(theta1)
        theta2 = np.random.rand(args['num_dp_per_client'])*2*np.pi
        theta2 = theta1.view(1,len(theta2))

        Y2 = gsigma*Q@gcs.T@torch.cos(theta2)+lsigma*Q@lcs[1].T@torch.sin(theta2)
        Y=[Y1,Y2]
        #Y,sn = add_sparse_noise(Y,args['sparsity'],100)
        #Y1,sn1 = add_sparse_noise(Y1,args['sparsity'],100)

        

        Ugt = Q@gcs.T
        vgst = [torch.cos(theta1).T*gsigma,torch.cos(theta2).T*gsigma]
        ulst = [Q@lcs[0].T,Q@lcs[1].T]
        vlst = [torch.sin(theta1).T*lsigma,torch.sin(theta2).T*lsigma]

        args["global_recovery_error"] = lambda x: g_recovery([Ugt,vgst,ulst,vlst],x)
        args["local_recovery_error"] = lambda x: l_recovery([Ugt,vgst,ulst,vlst],x)
        #args["sparse_recovery_error"] = lambda x: s_recovery([Ugt,vgst,ulst,vlst,Snoise],x)
        
        #Ug,Vg,Ul,Vl = lg_matrix_factorization(Y, args)
        Ug,Vg,Ul,Vl,S = heterogeneous_matrix_factorization_subgd(Y, args,verbose=20)
      
    
  
    def synthetic_nonoise(inputargs):
        args = {
            "ngc":3,
            "nlc":3,
            "optim":"SGD",
            "lr":0.1,
            "epochs":1000,
            "beta":1e-1,
            "outer_epochs":1000,
            "seed":100,
            "lbd_s":1000,
            "lbd_s_outer":100,
            "verbose":2,
            "rho":0.99,
            "wd":0,
            #"epsilon":1e-8,
            "tensorboard":0,
            "normalize_inner_loop":0,          
        }
        args['sparsity'] = inputargs['sparsity']
        print(args)
        d = 120
        ndt = 100
        N = 100
        #Ugt = torch.randn(d,args["ngc"])/np.sqrt(d)
        device = "cuda"

        Ugt = torch.zeros(d,args["ngc"]).to(device)
        for i in range(args["ngc"]):
            Ugt[i,i]=1#0

        Q, R = torch.linalg.qr(torch.randn(d,d))    
        Q = Q.to(device)
        Ult = torch.zeros(N, d, args["nlc"]).to(device)
        ter_id = 119
        ini_id = 5
        sparsity = int(args["sparsity"])
        for clid in range(N):
            for cpid in range(args["nlc"]):
                Ult[clid, ((cpid + clid) % (ter_id - ini_id + 1) + ini_id), cpid] = 1
                # res[clid, cpid, (cpid)%(ter_id-ini_id+1)+ini_id] = 1
        with torch.no_grad():
            Ugt = Q@Ugt
            for i in range(N):
                Ult[i,:,:] = Q@Ult[i,:,:]
        import torch.nn.functional as F
        #TORCH.NN.UTILS.PARAMETRIZATIONS.ORTHOGONAL
        pavg = sum(Ult[i,:,:]@Ult[i,:,:].T for i in range(N))/N
        up,sp,vhp = torch.linalg.svd(pavg)
        #print(sp)
        print("theta is %.6f"%(1-sp[0]))
        #return
        
        Y = []
        ulst = []
        vgst = []
        vlst = []
        for i in range(N):
            Vgt = torch.randn(ndt,args["ngc"]).to(device)#*100
            Qvg,R = torch.linalg.qr(Vgt)
            Vgt = Qvg
            #Ult = torch.randn(d,args["nlc"])
            Ult[i,:,:] -= Ugt @ torch.inverse(Ugt.T@Ugt) @ Ugt.T @ Ult[i,:,:]
            #Ult = F.normalize(Ult, p=2, dim=1)
            ulst.append(Ult[i,:,:])
            #print(Ult[i,:,:])
            Vlt = torch.randn(ndt,args["nlc"]).to(device)#*100
            Vlt -= Vgt @ torch.inverse(Vgt.T@Vgt) @ Vgt.T@Vlt
            Qvl,R = torch.linalg.qr(Vlt)
            Vlt = Qvl
            
            ygpart = Ugt@Vgt.T
            

            yipart = Ult[i,:,:]@Vlt.T
            
            
            Y.append(ygpart+yipart)
            #Y.append(Ugt@Vgt.T+Ult[i,:,:]@Vlt.T)
            vgst.append(Vgt)
            vlst.append(Vlt)
        with torch.no_grad():
            args["lbd_s_outer"]=torch.max(torch.abs(Ugt@Vgt.T))*2
            print(args["lbd_s_outer"])

        args["global_subspace_err_metric"]= lambda x: subspace_error(Ugt,x)
        args["local_subspace_err_metric"] = lambda x: sum([subspace_error(ulst[i],x[i]) for i in range(N)])/N
        
        #Y = torchimgpro.load_girl_data()
        #N = len(Y)
        #Y,Snoise = add_sparse_noise(Y,args['sparsity'],100)
        
        args["global_recovery_error"] = lambda x: g_recovery([Ugt,vgst,ulst,vlst],x,'f')
        args["local_recovery_error"] = lambda x: l_recovery([Ugt,vgst,ulst,vlst],x,'f')
        #args["sparse_recovery_error"] = lambda x: s_recovery([Ugt,vgst,ulst,vlst,Snoise],x)
       
        
        #Ug,Vg,Ul,Vl = lg_matrix_factorization(Y, args)
        Ug,Vg,Ul,Vl = heterpgeneous_matrix_factorization(Y,args,verbose=20)
   


    def video_reshuffle_example(inputargs):
        args = {
            "ngc":20,
            "nlc":100,
            "optim":"SGD",
            "lr":0.0000005,#0.005 is the best with normalization
            "epochs":5000,
            "outer_epochs":100,
            "seed":100,
            "lbd_s":1e4,
            "rho":0.95,
            "beta":1e-2,
            "wd":0,
            "normalize":0,
            "normalize_inner_loop":0,
            "columnnorm":0,
            "momentum":0,
            "reshuffle":1,
            "verbose":1,
            "device":"cuda"
        }
        '''
        # Best parameter for the car data under Fourier transformation:        
        args = {
            "ngc":200,
            "nlc":200,
            "optim":"SGD",
            "lr":0.01,
            "epochs":200,
            "seed":100,
            "lbd_s":1e4,
            "wd":0,
            "normalize":1,
            "columnnorm":1,
            "momentum":1,
            "reshuffle":0,
        }
        
        # Best parameter for the car data:
        1. do not transpose the flat figure
        2. args = {
            "ngc":60,
            "nlc":100,
            "optim":"SGD",
            "lr":0.01,
            "epochs":500,
            "seed":100,
            "lbd_s":"auto",
        }
        '''
        print(args)

        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        import torchimgpro 
        Y = torchimgpro.load_girl_data(args)
        #Y = Y[:5]
        if args["momentum"]:
            Y = torchimgpro.position2momentum(Y)
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
        #print(Y[0])
        print(Y[0].shape)
        maxelement = torch.max(torch.cat(Y).abs()).item()
        minelement = torch.min(torch.cat(Y).abs()).item()
        Y,Snoise = add_sparse_noise(Y,0.01,500)
        args['lbd_s_outer'] = 250
        norms = []
        transmat = []
        from sqrtm import sqrtm
        if args["normalize"]:
            print("normalize data")
            with torch.no_grad():
                for i in range(N):
                    ni = []
                    if args["columnnorm"]:
                        for j in range(len(Y[i])):
                            normy = Y[i][j].norm()
                            Y[i][j] /= normy
                            ni.append(normy)
                        norms.append(ni)
                    else:
                        for j in range(len(Y[i][0])):
                            normy = Y[i][:,j].norm()
                            Y[i][:,j] /= normy
                            ni.append(normy)
                        norms.append(ni)
                        '''
                        transmati = sqrtm(Y[i].T@Y[i])

                        Y[i] = Y[i]@torch.inverse(transmati)
                        transmat.append(transmati)
                        '''
            

        print(Y[0])
        args["sparse_recovery_error"] = lambda x: s_recovery([None,None,None,None,Snoise],x)

        #Ug,Vg,Ul,Vl = lg_matrix_factorization(Y, args)
        Ug,Vg,Ul,Vl,S = twoloop_matrix_factorization(Y,args)
        #Ug,Vg,Ul,Vl,S = heterogeneous_matrix_factorization_subgd(Y, args)

        reconstruct_bg = [(Ug[i].lin_mat@Vg[i].lin_mat.T) for i in range(N)]
        reconstruct_cat = [(Ul[i].lin_mat@Vl[i].lin_mat.T) for i in range(N)]
        '''
        for i in range(len(Y)):
                for j in range(len(Y[i])):
                    reconstruct_bg[i][j]*=np.linalg.norm(Y[i][j])
                    reconstruct_cat[i][j]*=np.linalg.norm(Y[i][j])
                    #Y[i][j] *=
        '''

        folder = r'dcpdframes/'
        with torch.no_grad():
            for i in range(N):
                print("saving figure [%s/%s]"%(i,N))
            
                if args["normalize"]:
                    if args["columnnorm"]:
                        for j in range(len(reconstruct_bg[i])):
                            reconstruct_bg[i][j] *= norms[i][j]
                            reconstruct_cat[i][j] *= norms[i][j]
                            Y[i][j] *= norms[i][j]
                            S[i].lin_mat[j] *= norms[i][j]
                    else:
                        for j in range(len(reconstruct_bg[i][0])):
                            reconstruct_bg[i][:,j] *= norms[i][j]
                            reconstruct_cat[i][:,j] *= norms[i][j]
                            Y[i][:,j] *= norms[i][j]
                            S[i].lin_mat[:,j] *= norms[i][j]
                        '''
                        reconstruct_bg[i] = reconstruct_bg[i]@transmat[i]
                        reconstruct_cat[i]= reconstruct_cat[i]@transmat[i]
                        Y[i] = Y[i]@transmat[i]
                        z  = S[i].lin_mat@transmat[i]
                        S[i].lin_mat *= 0
                        S[i].lin_mat += z
                        '''
                
                original = Y[i].detach().cpu().numpy()
                torchimgpro.show_save(original,folder+'original_%s.png'%i,maxelement,minelement,args=args)
                #plt.imshow(original, cmap='gray')
                #plt.axis('off')
                #plt.savefig(folder+'original_%s.png'%i, bbox_inches='tight')

                reconstruct_bg[i] = reconstruct_bg[i].detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_bg[i],folder+'recons_bg_%s.png'%i,args=args)
               
                reconstruct_cat[i] = reconstruct_cat[i].detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_cat[i],folder+'recons_cat_%s.png'%i,args=args)

             
                reconstruct_noise = S[i].lin_mat.detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_noise,folder+'recons_noise_%s.png'%i,args=args)

    def video_nips(inputargs):
        args = {
            "ngc":20,
            "nlc":100,
            "optim":"SGD",
            "lr":0.000005,#0.005 is the best with normalization
            "epochs":5000,
            "outer_epochs":1,
            "seed":100,
            "lbd_s":1e4,
            "rho":0.95,
            "beta":1e-2,
            "wd":0,
            "normalize":0,
            "normalize_inner_loop":0,
            "columnnorm":0,
            "momentum":0,
            "reshuffle":1,
            "verbose":1,
            "device":"cuda"
        }
        '''
        # Best parameter for the car data under Fourier transformation:        
        args = {
            "ngc":200,
            "nlc":200,
            "optim":"SGD",
            "lr":0.01,
            "epochs":200,
            "seed":100,
            "lbd_s":1e4,
            "wd":0,
            "normalize":1,
            "columnnorm":1,
            "momentum":1,
            "reshuffle":0,
        }
        
        # Best parameter for the car data:
        1. do not transpose the flat figure
        2. args = {
            "ngc":60,
            "nlc":100,
            "optim":"SGD",
            "lr":0.01,
            "epochs":500,
            "seed":100,
            "lbd_s":"auto",
        }
        '''
        print(args)

        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        import torchimgpro 
        Y = torchimgpro.load_girl_data(args)
        #Y = Y[:5]
        if args["momentum"]:
            Y = torchimgpro.position2momentum(Y)
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
        #print(Y[0])
        print(Y[0].shape)
        maxelement = torch.max(torch.cat(Y).abs()).item()
        minelement = torch.min(torch.cat(Y).abs()).item()
        #Y,Snoise = add_sparse_noise(Y,0.01,500)
        args['lbd_s_outer'] = 250
        norms = []
        transmat = []
        from sqrtm import sqrtm
        if args["normalize"]:
            print("normalize data")
            with torch.no_grad():
                for i in range(N):
                    ni = []
                    if args["columnnorm"]:
                        for j in range(len(Y[i])):
                            normy = Y[i][j].norm()
                            Y[i][j] /= normy
                            ni.append(normy)
                        norms.append(ni)
                    else:
                        for j in range(len(Y[i][0])):
                            normy = Y[i][:,j].norm()
                            Y[i][:,j] /= normy
                            ni.append(normy)
                        norms.append(ni)
                        '''
                        transmati = sqrtm(Y[i].T@Y[i])

                        Y[i] = Y[i]@torch.inverse(transmati)
                        transmat.append(transmati)
                        '''
            

        print(Y[0])
        #args["sparse_recovery_error"] = lambda x: s_recovery([None,None,None,None,Snoise],x)

        #Ug,Vg,Ul,Vl = lg_matrix_factorization(Y, args)
        Ug,Vg,Ul,Vl,S = twoloop_matrix_factorization(Y,args)
        #Ug,Vg,Ul,Vl,S = heterogeneous_matrix_factorization_subgd(Y, args)

        reconstruct_bg = [(Ug[i].lin_mat@Vg[i].lin_mat.T) for i in range(N)]
        reconstruct_cat = [(Ul[i].lin_mat@Vl[i].lin_mat.T) for i in range(N)]
        '''
        for i in range(len(Y)):
                for j in range(len(Y[i])):
                    reconstruct_bg[i][j]*=np.linalg.norm(Y[i][j])
                    reconstruct_cat[i][j]*=np.linalg.norm(Y[i][j])
                    #Y[i][j] *=
        '''

        folder = r'dcpdframes/'
        with torch.no_grad():
            for i in range(N):
                print("saving figure [%s/%s]"%(i,N))
            
                if args["normalize"]:
                    if args["columnnorm"]:
                        for j in range(len(reconstruct_bg[i])):
                            reconstruct_bg[i][j] *= norms[i][j]
                            reconstruct_cat[i][j] *= norms[i][j]
                            Y[i][j] *= norms[i][j]
                            S[i].lin_mat[j] *= norms[i][j]
                    else:
                        for j in range(len(reconstruct_bg[i][0])):
                            reconstruct_bg[i][:,j] *= norms[i][j]
                            reconstruct_cat[i][:,j] *= norms[i][j]
                            Y[i][:,j] *= norms[i][j]
                            S[i].lin_mat[:,j] *= norms[i][j]
                        '''
                        reconstruct_bg[i] = reconstruct_bg[i]@transmat[i]
                        reconstruct_cat[i]= reconstruct_cat[i]@transmat[i]
                        Y[i] = Y[i]@transmat[i]
                        z  = S[i].lin_mat@transmat[i]
                        S[i].lin_mat *= 0
                        S[i].lin_mat += z
                        '''
                
                original = Y[i].detach().cpu().numpy()
                torchimgpro.show_save(original,folder+'original_%s.png'%i,maxelement,minelement,args=args)
                #plt.imshow(original, cmap='gray')
                #plt.axis('off')
                #plt.savefig(folder+'original_%s.png'%i, bbox_inches='tight')

                reconstruct_bg[i] = reconstruct_bg[i].detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_bg[i],folder+'recons_bg_%s.png'%i,args=args)
               
                reconstruct_cat[i] = reconstruct_cat[i].detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_cat[i],folder+'recons_cat_%s.png'%i,args=args)

             
                reconstruct_noise = S[i].lin_mat.detach().cpu().numpy()
                torchimgpro.show_save(reconstruct_noise,folder+'recons_noise_%s.png'%i,args=args)

 
    def email(inputargs):
        args = {
            "ngc":50,
            "nlc":10,
            "optim":"SGD",
            "lr":0.005,
            "beta":0.01,
            "epochs":1000,
            "outer_epochs":1,
            "seed":100,
            "lbd_s":1000,
            "lbd_s_outer":1.2,
            "verbose":2,
            "rho":0.99,
            "wd":0,
            "tensorboard":0,
            "normalize_inner_loop":0,    
        }
        import emailprocess as ep

        Y = ep.loademail("cuda")   
        Y = [Y[i] for i in range(len(Y)) if torch.sum(torch.abs(Y[i]))>40] 
        print("%s samples"%len(Y))
        N = len(Y)

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
        Ug,Vg,Ul,Vl,S = heterogeneous_matrix_factorization_subgd(Ytrain,args)
        terr = ep.test_err(Ytest,Ug,Vg,Ul,Vl,test2full,full2train,train2full,prevtrain,nexttrain)
        print('hmf: test error %.4f'%terr)


        sumtrain = sum(Ytrain)
        upool,spool,vhpool = torch.svd(sumtrain)
        upool = upool[:,:(args['ngc']+args['nlc'])]
        err = sum([torch.norm(yi - upool@upool.T@yi) for yi in Ytest])/len(Ytest)
        print('pooled mf: test error %.4f'%err)
        

        ep.plotall(Y,Ug,Vg,Ul,Vl)
        return
      
    def stock(inputargs):
        args = {
            "ngc":10,
            "nlc":10,
            "optim":"SGD",
            "lr":0.001,
            "beta":0.01,
            "epochs":10000,
            "outer_epochs":1,
            "seed":100,
            "lbd_s":1000,
            "lbd_s_outer":1.2,
            "verbose":2,
            "rho":0.99,
            "wd":0,
            "tensorboard":0,
            "normalize_inner_loop":0,    
        }
        import stockprocess as sp

        Y,rawdf = sp.loadstocks("cuda")   
        #Y = [Y[i] for i in range(len(Y)) if torch.sum(torch.abs(Y[i]))>40] 
        print("%s samples"%len(Y))
        print(Y[0].shape)
        N = len(Y)
        Ug,Vg,Ul,Vl,S = heterogeneous_matrix_factorization_subgd(Y,args)       
        
    
        lfac = []
        
        for i in range(N):
            #si = torch.max(torch.abs(Ul[i].lin_mat@Vl[i].lin_mat.T)).cpu().item()
            si = torch.sum(torch.abs(Ul[i].lin_mat@Vl[i].lin_mat.T).cpu(),dim=0)#.view(-1,1)
            
            lfac.append(si)
        sp.plotall(rawdf,lfac)
        return 
     

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='heterogeneous matrix factorization')
    parser.add_argument('--dataset', type=str, default="synthetic_twoloop_example")
    parser.add_argument('--algorithm', type=str, default="dgd")
    parser.add_argument('--logoutput', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--d', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=150)

    parser.add_argument('--num_client', type=int, default=100)
    parser.add_argument('--nlc', type=int, default=1)
    parser.add_argument('--ngc', type=int, default=1)
    parser.add_argument('--sparsity', type=float, default=0.)
    parser.add_argument('--toytheta', type=float, default=60.)


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
    '''
    if True:
        import os
        from misc import Tee
        import time
        import sys
        output_dir = 'outputs/'
        jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())
        output_dir += jour
        os.makedirs(output_dir, exist_ok=True)
        sys.stdout = Tee(os.path.join(output_dir, 'out.txt'))
    '''
    #movie_simple()
    #movie_personalized()
    #video_example()
    #synthetic_example_tensor()
    #synthetic_example()
    #qsr_personalized()
    #video_super_robust()  
    #video_tensor_robust_example()  
    #img_example()
