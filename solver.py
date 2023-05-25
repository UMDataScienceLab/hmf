#basic libary
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import time

class Matrix(nn.Module):## Matrix model, bypasses many issues in tensor copying
    def __init__(self, row_d, col_d,rto=0.01,device="cpu"):
        super(Matrix, self).__init__()
        self.lin_mat = nn.Parameter(rto*torch.randn(row_d, col_d, requires_grad=True,device=device))

    def forward(self,x):
        return torch.matmul(self.lin_mat, x)

def subspace_error(U,V):
    with torch.no_grad():    
        pu = U@torch.linalg.pinv(U.T@U)@U.T
        #pu = U.lin_mat@torch.inverse(U.lin_mat.T@U.lin_mat)@U.lin_mat.T
        pv = V.lin_mat@torch.linalg.pinv(V.lin_mat.T@V.lin_mat)@V.lin_mat.T
        return torch.norm(pu-pv).item()

def g_recovery(truelist, estimate, norm='inf'):
    ugtrue = truelist[0]
    vgtrue = truelist[1]

    uge = estimate[0]
    vge = estimate[1]
    if isinstance(uge, list):
        N = len(uge)
        alliters = list(range(N))
    else:
        alliters = uge.keys()
        N = len(alliters)
    res = 0.
    with torch.no_grad():
        for k in alliters:
            if norm  == 'inf':
                res += torch.max(torch.abs(ugtrue@vgtrue[k].T-uge[k].lin_mat@vge[k].lin_mat.T))
            else:
                res += torch.norm(torch.abs(ugtrue@vgtrue[k].T-uge[k].lin_mat@vge[k].lin_mat.T))

        return res/N

def l_recovery(truelist, estimate, norm='inf'):
    ultrue = truelist[2]
    vltrue = truelist[3]

    ule = estimate[2]
    vle = estimate[3]
    if isinstance(ule, list):
        N = len(ule)
        alliters = list(range(N))
    else:
        alliters = ule.keys()
        N = len(alliters)
    res = 0.
    with torch.no_grad():
        for k in alliters:
            if norm == 'inf':
                res += torch.max(torch.abs(ultrue[k]@vltrue[k].T-ule[k].lin_mat@vle[k].lin_mat.T))
            else:
                res += torch.norm(torch.abs(ultrue[k]@vltrue[k].T-ule[k].lin_mat@vle[k].lin_mat.T))
            
    return res/N



def heterpgeneous_matrix_factorization(Yin,args,initialization=[],verbose=1):
    
    if isinstance(Yin, list):
        N = len(Yin)
        alliters = list(range(N))
    else:
        alliters = Yin.keys()
        N = len(alliters)
    
  
    Y = Yin
    n2dict = {}
    lastloss = 1e10
    for y in alliters:
        (n1,n2dict[y]) = Y[y].shape
        

    if isinstance(args["nlc"], list):
        nlclst = args["nlc"]
    else:
        nlclst = [args["nlc"] for i in range(N)]
    if len(initialization) == 0:
        Ug = {k:Matrix(n1,args["ngc"],device=Y[k].device) for k in alliters}
        Ug_avg = Matrix(n1,args["ngc"],device=Y[y].device)
        Vg = {k:Matrix(n2dict[k],args["ngc"],device=Y[k].device) for k in alliters}
        Ul = {k:Matrix(n1,nlclst[k],device=Y[k].device) for k in alliters}
        Vl = {k:Matrix(n2dict[k],nlclst[k],device=Y[k].device) for k in alliters}
    else:
        Ug = copy.deepcopy(initialization[0])
        for ugi in Ug:
            Ug_avg = copy.deepcopy(Ug[ugi])
            break
        Vg = copy.deepcopy(initialization[1])
        Ul = copy.deepcopy(initialization[2])
        Vl = copy.deepcopy(initialization[3])

    parlist = {i:list(Ug[i].parameters())+list(Vg[i].parameters())+   list(Ul[i].parameters())+list(Vl[i].parameters()) for i in alliters}
    if args["optim"] == "SGD":
        optim = {k:torch.optim.SGD(parlist[k], lr=args["lr"], weight_decay=args["wd"]) for k in alliters}
    else:
        raise Exception("Error: The optimizor %s is not impkemented for hmf."%args['optim'])
    
    for n in range(args["epochs"]):
        time_start = time.time() 
        
        tot_loss = 0
        tot_reg = 0
        tot_ureg = 0
        for i in alliters:
            #gradient descent step
            pred = Ug[i].lin_mat@Vg[i].lin_mat.T+ Ul[i].lin_mat@Vl[i].lin_mat.T 
            regi = (torch.sum((Ug[i].lin_mat.T@Ug[i].lin_mat-torch.eye(args["ngc"],device=Y[i].device))**2)+torch.sum((Ul[i].lin_mat.T@Ul[i].lin_mat-torch.eye(nlclst[i],device=Y[i].device))**2))
            
            lossi = torch.sum((pred-Y[i])**2)+args["beta"]*regi#print(n1,n2dict[i]))**2)+torch.sum(()**2))
            optim[i].zero_grad()
            lossi.backward()
            optim[i].step()
            
            tot_loss += lossi.item()
            tot_ureg += regi.item()

        with torch.no_grad():
            # averaging step
            Ug_avg.lin_mat *= 0
            Ug_avg.lin_mat += sum([Ug[i].lin_mat for i in alliters])/N
            
            #correction step
            pj0 = torch.inverse(Ug_avg.lin_mat.T@Ug_avg.lin_mat)@Ug_avg.lin_mat.T
            projection = Ug_avg.lin_mat@pj0
            for i in alliters:
                Ug[i].lin_mat *= 0
                Ug[i].lin_mat += Ug_avg.lin_mat
                #Ul[i].lin_mat -= Ug_avg.lin_mat@torch.inverse(Ug_avg.lin_mat.T@Ug_avg.lin_mat)@(Ug_avg.lin_mat.T@Ul[i].lin_mat)
                Vg[i].lin_mat += Vl[i].lin_mat@Ul[i].lin_mat.T@pj0.T
                Ul[i].lin_mat -= projection@Ul[i].lin_mat
        
   
        tot_loss /= N
        tot_reg /= N
        tot_ureg /= N
        if tot_loss > lastloss:
            print("WARNING")
            print("loss is increased, decrease the stepsize")
            first = True
            for j in alliters:
                for g in optim[j].param_groups:
                    g['lr'] *= np.exp(-1)
                    if first:
                        print('new stepsize: %.8f'%g['lr'])    
                    first = False  
        if "epsilon" in args and "break_early" in args:
            if (lastloss - tot_loss)/ tot_loss < args["epsilon"]:# and tot_loss <1:
                print("ttloss %.4f"%tot_loss)
                print("inner loop converged in %s iterations"%n)
                break
            else:
                if n %(args["epochs"]//10) == 0:
                    print((lastloss - tot_loss)/ tot_loss)
            #lastloss = tot_loss
        lastloss = tot_loss

        output = "[%s/%s], loss %.6f, reg %.6f, ureg %.6f"%(n,args["epochs"],tot_loss, tot_reg, tot_ureg)
        if "global_subspace_err_metric" in args.keys():
            output += " gserr %s "%args["global_subspace_err_metric"](Ug_avg)
        if "local_subspace_err_metric" in args.keys():
            output += " lserr %s "%args["local_subspace_err_metric"](Ul)
        if "global_recovery_error" in args.keys():
            output += " g_recovery_err %.8f, "%args["global_recovery_error"]([Ug,Vg,Ul,Vl])
        if "local_recovery_error" in args.keys():
            output += " l_recovery_err %.8f, "%args["local_recovery_error"]([Ug,Vg,Ul,Vl])
        
        time_end = time.time()
        if (verbose>0.5 and n%(args["epochs"]//10)==0) or verbose >10:
            print(output+", time %s"%(time_end-time_start))

    return Ug, Vg, Ul, Vl

