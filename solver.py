#basic libary
import copy
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import time
from sklearn.linear_model import HuberRegressor, LinearRegression
from scipy.optimize import minimize
from joblib import Parallel, delayed


class Matrix(nn.Module):## Matrix model, bypasses many issues in tensor copying
    def __init__(self, row_d, col_d,rto=0.01,device="cpu"):
        super(Matrix, self).__init__()
        self.lin_mat = nn.Parameter(rto*torch.randn(row_d, col_d, requires_grad=True,device=device))

    def forward(self,x):
        return torch.matmul(self.lin_mat, x)

def subspace_error(U,V):
    with torch.no_grad():    
        pu = U@torch.linalg.pinv(U.T@U)@U.T
        if type(V) == torch.nn.parameter.Parameter:
            pv = V@torch.linalg.pinv(V.T@V)@V.T
        else:
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


def soft(z, lam):  
    with torch.no_grad():
        lala = z
        lala[torch.abs(lala)<lam] = 0
        #shrink = lam*torch.sign(z)
        #shrink[torch.abs(lala)<lam] = 0
        #lala -= shrink
        return lala
        
        m=torch.abs(z)-lam
        m[m<0] = 0 
        return torch.sign(z)*m
    
def retract(mata):
   u, s, v = torch.svd(mata)    
   s *= 0
   s += 1   
   return  torch.mm(torch.mm(u, torch.diag(s)), v.t())

def rob_svd(M,r=1):
    def rob_svd_top_0(A,niter=10,printupdate=False):
        Ac = copy.deepcopy(A)
        Ac[Ac>1.35] = 1.35
        Ac[Ac<-1.35] = -1.35
        u,s,vh = np.linalg.svd(Ac)
        m,n = A.shape
        u0 = u[0:1].T*np.sqrt(s[0])
        v0 = vh[:,0:1]*np.sqrt(s[0])
        #print(u0.shape)
        #print(v0.shape)
        #assert False
        hr = HuberRegressor()
        for i in range(niter):
            ulast = copy.deepcopy(u0)
            for j in range(n):
                hr.fit(u0, A[:,j])
                v0[j,0] = hr.coef_[0]
            for k in range(m):
                hr.fit(v0, A.T[:,k])
                u0[k,0] = hr.coef_[0]
            #if np.linalg.norm(u0-ulast) < 1e-2:
            #    break
            if printupdate:
                print(i,np.linalg.norm(u0-ulast))            
        return u0,v0
    def rob_svd_top_0_1(A,niter=10,printupdate=False):
        Ac = copy.deepcopy(A)
        Ac[Ac>1.35] = 1.35
        Ac[Ac<-1.35] = -1.35
        u,s,vh = np.linalg.svd(Ac)
        m,n = A.shape
        u0 = u[0:1].T*np.sqrt(s[0])
        v0 = vh[:,0:1]*np.sqrt(s[0])
        #print(u0.shape)
        #print(v0.shape)
        #assert False
        def hr_fit(y,x):
            hr = HuberRegressor()
            hr.fit(y,x)
            return hr.coef_[0]

        for i in range(niter):
            ulast = copy.deepcopy(u0)
            models = Parallel(n_jobs=-1)(delayed(hr_fit)(u0,A[:,j]) for j in range(n))
            
            v0 = np.array(models).reshape(n,1)
            models = Parallel(n_jobs=-1)(delayed(hr_fit)(v0,A.T[:,j]) for j in range(m))

            u0 = np.array(models).reshape(m,1)
            #if np.linalg.norm(u0-ulast) < 1e-2:
            #    break
            if printupdate:
                print(i,np.linalg.norm(u0-ulast))            
        return u0,v0
    
    def rob_svd_top(A,niter=10,printupdate=False):
        Ac = copy.deepcopy(A)
        Ac[Ac>1.35] = 1.35
        Ac[Ac<-1.35] = -1.35
        try:
            u,s,vh = np.linalg.svd(Ac)
        except:
            print("Error happened")
            print(Ac)
            print(A)
            assert False
        m,n = A.shape
        u0 = u[0:1].T*np.sqrt(s[0])
        v0 = vh[:,0:1]*np.sqrt(s[0])
        #print(u0.shape)
        #print(v0.shape)
        #assert False
        theta = 1.35
        for i in range(niter):
            ulast = copy.deepcopy(u0)
            coeffmat = A - u0@v0.T
            coeffmat[coeffmat > 2*theta] = 2*theta
            coeffmat[coeffmat < -2*theta] = -2*theta
            divisor = A - u0@v0.T
            divisor[np.abs(divisor)<1e-8] = 1e-8
            psi = coeffmat/divisor          
            Aajussted = A*psi

            u0 = Aajussted@v0/(psi@(v0**2))
            coeffmat = A - u0@v0.T
            coeffmat[coeffmat > theta] = theta
            coeffmat[coeffmat < -theta] = -theta
            divisor = A - u0@v0.T
            divisor[np.abs(divisor)<1e-8] = 1e-8
            psi = coeffmat/divisor             
            Aajussted = A*psi

            v0 = Aajussted.T@u0/(psi.T@(u0**2))

            #print(i,np.linalg.norm(u0-ulast))
        #print(u0)
        return u0,v0
    def rob_svd_top_3(A,niter=10,printupdate=False):
        u,s,vh = np.linalg.svd(A)
        m,n = A.shape
        u0 = u[0:1].T*np.sqrt(s[0])
        v0 = vh[:,0:1]*np.sqrt(s[0])
        def huber_loss(A, u, v, alpha=0.0001):
            diff = A - u@v.T
            theta = 1.35
            huber_diff = np.where(np.abs(diff) <= theta, 0.5 * diff**2, np.abs(diff)*theta - theta**2*0.5)
            return np.sum(huber_diff) + alpha*(np.sum(u**2)+np.sum(v**2))
        for i in range(niter):
            ulast = copy.deepcopy(u0)

            lossv = lambda x: huber_loss(A,u0,x.reshape(n,1))
            res = minimize(lossv, v0.flatten(), method='L-BFGS-B')
            v0 = res.x.reshape(n,1)

            lossu = lambda x: huber_loss(A,x.reshape(m,1),v0)
            res = minimize(lossu, u0.flatten(), method='L-BFGS-B')
            u0 = res.x.reshape(m,1)
                     
        return u0,v0
    
    def rob_svd_top_4(A):
        def huber_loss(x, delta):
            return np.where(np.abs(x) <= delta, x**2 / 2, delta * (np.abs(x) - delta / 2))

        def huber_loss_grad(x, delta):
            return np.where(np.abs(x) <= delta, x, delta * np.sign(x))

        def huber_loss_fun(params, A, delta):
            u = params[:A.shape[0]]
            v = params[A.shape[0]:]
            res = np.outer(u, v)
            loss = np.sum(huber_loss(A - res, delta))
            return loss

        def huber_loss_grad_fun(params, A, delta):
            u = params[:A.shape[0]]
            v = params[A.shape[0]:]
            res = np.outer(u, v)
            
            hlg = huber_loss_grad(A - res, delta)
            grad_u = hlg@ v.reshape(A.shape[1],1)
            grad_v = hlg.T @ u.reshape(A.shape[0],1)
            
            return np.concatenate([grad_u[:,0], grad_v[:,0]])

        
        delta = 1.35

        # Define the objective function for the optimizer
        objective = lambda params: huber_loss_fun(params, A, delta)

        # Define the gradient of the objective function for the optimizer
        gradient = lambda params: huber_loss_grad_fun(params, A, delta)

        # Concatenate u and v
        Ac = copy.deepcopy(A)
        Ac[Ac>1.35] = 1.35
        Ac[Ac<-1.35] = -1.35
        u,s,vh = np.linalg.svd(Ac)
        m,n = A.shape
        u0 = u[0]*np.sqrt(s[0])
        v0 = vh[:,0]*np.sqrt(s[0])
        params = np.concatenate([u0, v0])

        # Minimize the objective function using L-BFGS-B method
        print(objective(params))
        print(objective(params*0))
        result = minimize(objective, params, jac=gradient, method='L-BFGS-B')

        u0 = result.x[:m].reshape(m,1)
        v0 = result.x[m:].reshape(n,1)
        print(np.linalg.norm(u0@v0.T))
        return u0,v0

    A = copy.deepcopy(M)
    ulist = []
    vlist = []
    for rk in range(r):
        #print(rk)
        ui,vi = rob_svd_top(A)
        A = A - ui@vi.T

        ulist.append(ui)
        vlist.append(vi)

    return np.concatenate(ulist,axis=1), np.concatenate(vlist,axis=1)
  

def heterogeneous_matrix_factorization(Yin,args,initialization=[],verbose=1):
    
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
        

    if isinstance(args["r2"], list):
        nlclst = args["r2"]
    else:
        nlclst = [args["r2"] for i in range(N)]
    if len(initialization) == 0:
        Ug = {k:Matrix(n1,args["r1"],device=Y[k].device) for k in alliters}
        Ug_avg = Matrix(n1,args["r1"],device=Y[y].device)
        Vg = {k:Matrix(n2dict[k],args["r1"],device=Y[k].device) for k in alliters}
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
            regi = (torch.sum((Ug[i].lin_mat.T@Ug[i].lin_mat-torch.eye(args["r1"],device=Y[i].device))**2)+torch.sum((Ul[i].lin_mat.T@Ul[i].lin_mat-torch.eye(nlclst[i],device=Y[i].device))**2))
            
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


def perpca(Y, args,initialization=[]):
    (n1,n2) = Y[0].shape
    N = len(Y)
    if len(initialization) == 0:
        Ug = [torch.randn(n1,args["ngc"],dtype=Y[k].dtype).to(args['device'])*0.001 for k in range(N)]
        Ug_avg = torch.randn(n1,args["ngc"],dtype=Y[0].dtype).to(args['device'])*0.001
        Ul = [torch.randn(n1,args["nlc"],dtype=Y[k].dtype).to(args['device'])*0.001 for k in range(N)]
    else:
        Ug = [torch.randn(n1,args["ngc"],dtype=Y[k].dtype).to(args['device'])*0.001 for k in range(N)]
        Ug_avg = torch.randn(n1,args["ngc"],dtype=Y[0].dtype).to(args['device'])*0.001

        Ul = [torch.randn(n1,args["nlc"],dtype=Y[k].dtype).to(args['device'])*0.001 for k in range(N)]
        with torch.no_grad():
            for i in range(N):
                Ug[i] *= 0
                Ug[i] += initialization[0][i]
                Ug_avg *= 0
                Ug_avg += Ug[0]
               
                Ul[i] *= 0
                Ul[i] += initialization[1][i]

    #print(Ug[0])
    #print(Ug[0].dtype)
    #print(Y[0].dtype)
    #assert False
    minloss = 1000000
    noprogress = 0
    for n in range(args["epochs"]):
        time_start = time.time()
        tot_loss = 0
        for i in range(N):
            # correction
            with torch.no_grad():
                Ulcorrected = Ul[i] - Ug[i]@Ug[i].T@Ul[i]
                Ul[i] = retract(Ulcorrected)
                Ug[i] += args['lr'] * Y[i]@(Y[i].T@Ug[i])
                Ul[i] += args['lr'] * Y[i]@(Y[i].T@Ul[i])

                retracted = retract(torch.cat((Ug[i],Ul[i]),dim=1))

                Ug[i] = retracted[:,:args['ngc']]
                Ul[i] = retracted[:,args['ngc']:]

            tot_loss += torch.norm(Y[i].T@torch.cat((Ug[i],Ul[i]),dim=1))**2/(n1*n2)

        with torch.no_grad():
            Ug_avg *= 0
            Ug_avg += retract(sum([Ug[i] for i in range(N)])/N)
            for i in range(N):
                Ug[i] *= 0
                Ug[i] += Ug_avg  
        tot_loss /= N

        # if the loss is too small, break
        if "break_on_epsilon" in args.keys():
            if tot_loss < args["break_on_epsilon"]:
                break

        # if there is no progress for many iterations, break
        
        if tot_loss < minloss:
            minloss = tot_loss
            noprogress = 0
        else:
            noprogress += 1
        if 'noprogressthreshold' in args.keys():
            if noprogress > args['noprogressthreshold']:
                print("reduceing the stepsize, loss %.6f"%tot_loss)
                with torch.no_grad():
                    for k in range(N):
                        for i, param_group in enumerate(optim[k].param_groups):
                            param_group['lr']*= np.exp(-1)
                print(param_group['lr'])
                noprogress = 0
        time_end = time.time()
        output = "[%s/%s], loss %.6f, "%(n,args["epochs"],tot_loss)
        if "global_subspace_err_metric" in args.keys():
            output += " gserr %s "%args["global_subspace_err_metric"](Ug_avg)
        if "local_subspace_err_metric" in args.keys():
            output += " lserr %s "%args["local_subspace_err_metric"](Ul)
            output += " adderr %s "% (args["global_subspace_err_metric"](Ug_avg)+args["local_subspace_err_metric"](Ul))
        '''
        if "global_recovery_error" in args.keys():
            output += " g_recovery_err %.8f, "%args["global_recovery_error"]([Ug,Vg,Ul,Vl])
        if "local_recovery_error" in args.keys():
            output += " l_recovery_err %.8f, "%args["local_recovery_error"]([Ug,Vg,Ul,Vl])
        '''
        #time_end = time.time()
        if (args['verbose']>0.5 and n%(args["epochs"]//10+1)==0) or args['verbose'] >10 or n == args["epochs"]-1:
            print(output+", time %s"%(time_end-time_start))
        #print("[%s/%s], loss %s"%(n,args["epochs"],tot_loss))
    
    return Ug, Ul

def robustpca(Y, args,initialization=[]):
    (n1,n2) = Y[0].shape
    N = len(Y)
    Yallinone = torch.stack([Y[i].flatten() for i in range(len(Y))])
    lbd = args["lbd_s_outer"]

    L = copy.deepcopy(Yallinone)
    L *= 0
    for n in range(args["outer_epochs"]):
        time_start = time.time()
        with torch.no_grad():

            S = soft(Yallinone - L, lbd)
            tot_loss = 0
            lbd *= args["rho"]
        
            u, s, v = torch.svd(Yallinone - S)
            s[(args["ngc"]+args["nlc"]):] *= 0
            L = torch.mm(torch.mm(u, torch.diag(s)), v.t())
            tot_reg = torch.count_nonzero(S)
           
        tot_loss = torch.norm(Yallinone-L-S).item()
        output = "[%s/%s], loss %.6f, reg %.6f "%(n,args["outer_epochs"],tot_loss, tot_reg)
       
        time_end = time.time()
        if (args['verbose']>0.5 and n%(args["outer_epochs"]//10+1)==0) or args['verbose'] >10 or n == args["outer_epochs"]-1:
            print(output+", time %s"%(time_end-time_start))
        #print("[%s/%s], loss %s"%(n,args["epochs"],tot_loss))
    low_rank_part = [L[i].view(Y[i].shape) for i in range(len(Y))]
    sparse_part = [S[i].view(Y[i].shape) for i in range(len(Y))]

    return low_rank_part, sparse_part



def jive(Yin, args, initialization=[]):
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
        (n1, n2dict[y]) = Y[y].shape

    if isinstance(args["nlc"], list):
        nlclst = args["nlc"]
    else:
        nlclst = [args["nlc"] for i in range(N)]
    if len(initialization) == 0:
        J = {k: torch.randn((n1, n2dict[k]), device=Y[k].device) for k in alliters}
        A = {k: torch.randn((n1, n2dict[k]), device=Y[k].device)*0 for k in alliters}
    else:
        raise Exception('not implemented for non-standard initialization!')


    for n in range(args["epochs"]):
        time_start = time.time()

        tot_loss = 0
        tot_reg = 0

        with torch.no_grad():
            res_list = [Yin[k] - A[k] for k in alliters]
            res_concat = torch.cat(res_list, dim=1)
            u,s,vh = torch.svd(res_concat)
            projto = u[:,:args["ngc"]]
            s[args['ngc']:] *= 0
            res_concat = u@torch.diag_embed(s)@vh.T
            startid = 0
            for i in alliters:
                n1i, n2i = Yin[i].shape
                J[i] *= 0
                J[i] += res_concat[:, startid:(startid+n2i)]
                startid += n2i

            for i in alliters:
                # update individual parts
                resi = Yin[i] - J[i]
                u, s, vh = torch.svd(resi)
                s[args['nlc']:] *= 0
                A[i] *= 0
                A[i] += u @ torch.diag_embed(s) @ vh.T

            for i in alliters:
                tot_loss += torch.sum((Yin[i]-J[i]-A[i])**2)

        tot_loss /= N
        tot_reg /= N
        
        time_end = time.time()

        output = "[%s/%s], loss %.6f, " % (n, args["epochs"], tot_loss)
        if "global_subspace_err_metric" in args.keys():
            
            output += " gserr %s "%args["global_subspace_err_metric"](projto)

        if "local_subspace_err_metric" in args.keys():
            Ul = []
            for i in alliters:
                u, s, vh = torch.svd(A[i])
                Ul.append(u[:,:args['nlc']])

            output += " lserr %s "%args["local_subspace_err_metric"](Ul)
            output += " adderr %s "% (args["global_subspace_err_metric"](projto)+args["local_subspace_err_metric"](Ul))

        if "jive_global_recovery_error" in args.keys():
            output += " g_recovery_err %.8f, " % args["jive_global_recovery_error"]([J,A])
        if "jive_local_recovery_error" in args.keys():
            output += " l_recovery_err %.8f, " % args["jive_local_recovery_error"]([J,A])

        if (args['verbose'] > 0.5 and n % (args["epochs"] // 10) == 0) or args['verbose'] > 10 or n == args["epochs"]-1:
            print(output + ", time %s" % (time_end - time_start))

    return J, A



def robust_jive(Yin, args, initialization=[]):
    if isinstance(Yin, list):
        N = len(Yin)
        alliters = list(range(N))
    else:
        alliters = Yin.keys()
        N = len(alliters)

    n2dict = {}
    lastloss = 1e10
    for y in alliters:
        (n1, n2dict[y]) = Yin[y].shape

    if isinstance(args["nlc"], list):
        nlclst = args["nlc"]
    else:
        nlclst = [args["nlc"] for i in range(N)]
    if len(initialization) == 0:
        J = {k: torch.randn((n1, n2dict[k]), device=Yin[k].device) for k in alliters}
        A = {k: torch.randn((n1, n2dict[k]), device=Yin[k].device)*0 for k in alliters}
        E = {k: torch.randn((n1, n2dict[k]), device=Yin[k].device)*0 for k in alliters}
        F = {k: torch.randn((n1, n2dict[k]), device=Yin[k].device)*0 for k in alliters}
        R = {k: torch.randn((n1, n2dict[k]), device=Yin[k].device)*0 for k in alliters}
        Y = {k: torch.randn((n1, n2dict[k]), device=Yin[k].device)*0 for k in alliters}

    else:
        raise Exception('not implemented for non-standard initialization!')


    for n in range(args["epochs"]):
        time_start = time.time()

        tot_loss = 0
        tot_reg = 0

        with torch.no_grad():

            # update J
            res_list = [Yin[k] - A[k] -E[k] + F[k]/args['mu'] for k in alliters]
            res_concat = torch.cat(res_list, dim=1)
            u,s,vh = torch.svd(res_concat)
            s[args['ngc']:] *= 0

            projto = u[:,:args['ngc']]
            res_concat = u@torch.diag_embed(s)@vh.T
            startid = 0
            for i in alliters:
                n1i, n2i = Yin[i].shape
                J[i] *= 0
                J[i] += res_concat[:, startid:(startid+n2i)]
                startid += n2i
            
            # update A
            for i in alliters:
                resi = (Yin[i] - J[i] - E[i] + R[i] + (F[i]+Y[i])/args["mu"])/2
                resi = resi - projto@(projto.T@resi)
                #print('print shape')
                #print(resi.shape)
                #print(A[i].shape)
                A[i] *= 0
                A[i] += resi
            
            # update R
            for i in alliters:
                resi = A[i] - Y[i]/args["mu"]

                u, s, vh = torch.svd(resi)
                #s[args['nlc']:] *= 0
                s = (s-1/args["mu"]*torch.sign(s))*(torch.abs(s)>1/args["mu"])

                R[i] *= 0
                R[i] += u @ torch.diag_embed(s) @ vh.T

            # update E
            for i in alliters:
                resi = Yin[i] - J[i] - A[i] + F[i]/args["mu"]
                thresh = args["lbd"]/args["mu"]
                resi = (resi-thresh*torch.sign(resi))*(torch.abs(resi)>thresh)
                E[i] *= 0
                E[i] += resi
            
            # update langrangian multipliers
            for i in alliters:
                F[i] = F[i] + args["mu"]*(Yin[i] - J[i] - A[i] - E[i])
                Y[i] = Y[i] + args["mu"]*(R[i] - A[i])
            for i in alliters:
                tot_loss += (torch.norm(R[i],p='nuc')+args['lbd']*torch.norm(E[i],1))#(torch.abs(Yin[i]-J[i]-A[i]))

        tot_loss /= N
        tot_reg /= N

        output = "[%s/%s], loss %.6f, " % (n, args["epochs"], tot_loss)
        if "global_subspace_err_metric" in args.keys():
            output += " gserr %s "%args["global_subspace_err_metric"](projto)

        if "local_subspace_err_metric" in args.keys():
            Ul = []
            for i in alliters:
                u, s, vh = torch.svd(A[i])
                Ul.append(u[:,:args['nlc']])

            output += " lserr %s "%args["local_subspace_err_metric"](Ul)
            output += " adderr %s "% (args["global_subspace_err_metric"](projto)+args["local_subspace_err_metric"](Ul))

        if "rjive_global_recovery_error" in args.keys():
            output += " g_recovery_err %.8f, " % args["rjive_global_recovery_error"]([J,A])
        if "rjive_local_recovery_error" in args.keys():
            output += " l_recovery_err %.8f, " % args["rjive_local_recovery_error"]([J,A])
        if "rjive_e_recovery_error" in args.keys():
            output += " e_recovery_err %.8f, " % args["rjive_e_recovery_error"]([E])

        time_end = time.time()
        if (args['verbose'] > 0.5 and n % (args["epochs"] // 10) == 0) or args['verbose'] > 10 or n == args["epochs"]-1:
            print(output + ", time %s" % (time_end - time_start))

    return J, A, E


def rajive(Yin, args):
    if isinstance(Yin, list):
        N = len(Yin)
        alliters = list(range(N))
    else:
        alliters = Yin.keys()
        N = len(alliters)

    # phase 1: initial signal space extraction
    print("rajive, phase 1")
    utilde_list = []
    for i in range(N):
        print(i)
        uhat,vhat = rob_svd(Yin[i].cpu().numpy(),r=args["ngc"]+args["nlc"])
        utilde, vh = np.linalg.qr(uhat)
        utilde_list.append(utilde)

    # phase 2: score space segmentation
    print("rajive, phase 2")

    uconcat = np.concatenate(utilde_list,axis=1)
    ushared,vshared = rob_svd(uconcat,r=args["ngc"])
    ushared, vshared = np.linalg.qr(ushared)

    # phase 3: final decomposition 
    print("rajive, phase 3")

    J = []
    A = []
    E = []
    for i in range(N):
        yi = Yin[i].cpu().numpy()
        yijoint = ushared@ushared.T@yi
        ujhat,vjhat = rob_svd(yijoint,r=args["ngc"])
        J.append(torch.tensor(ujhat@vjhat.T,device=Yin[i].device))
        
        yiindiv = yi-yijoint
        uihat,vihat = rob_svd(yiindiv,r=args["nlc"])
        A.append(torch.tensor(uihat@vihat.T,device=Yin[i].device))
        E.append(torch.tensor(yi-ujhat@vjhat.T- uihat@vihat.T,device=Yin[i].device))

    
    return J, A, E



def heterogeneous_matrix_completion(Y, Ymask, args,initialization=[]):
    (n1,n2) = Y[0].shape
    N = len(Y)
    if len(initialization) == 0:
        Ug = [Parameter(torch.randn(n1,args["ngc"]).to(args['device'])*0.001) for k in range(N)]
        Ug_avg = Parameter(torch.randn(n1,args["ngc"]).to(args['device'])*0.001)
        Vg = [Parameter(torch.randn(n2,args["ngc"]).to(args['device'])*0.001) for k in range(N)]
        Ul = [Parameter(torch.randn(n1,args["nlc"]).to(args['device'])*0.001) for k in range(N)]
        Vl = [Parameter(torch.randn(n2,args["nlc"]).to(args['device'])*0.001) for k in range(N)]
    else:
        Ug = [Parameter(torch.randn(n1,args["ngc"]).to(args['device'])*0.001) for k in range(N)]
        Ug_avg = Parameter(torch.randn(n1,args["ngc"]).to(args['device'])*0.001)

        Vg = [Parameter(torch.randn(Y[k].size()[1],args["ngc"]).to(args['device'])*0.001) for k in range(N)]
        Ul = [Parameter(torch.randn(n1,args["nlc"]).to(args['device'])*0.001) for k in range(N)]
        Vl = [Parameter(torch.randn(Y[k].size()[1],args["nlc"]).to(args['device'])*0.001) for k in range(N)]
        with torch.no_grad():
            for i in range(N):
                Ug[i] *= 0
                Ug[i] += initialization[0][i]
                Ug_avg *= 0
                Ug_avg += Ug[0]
                Vg[i] *= 0
                Vg[i] += initialization[1][i]
                
                Ul[i] *= 0
                Ul[i] += initialization[2][i]
                

                Vl[i] *= 0
                Vl[i] += initialization[3][i]
 
    parlist = [[Ug[i]]+[Vg[i]]+
            [Ul[i]]+[Vl[i]] for i in range(N)]
    if args["optim"] == "SGD":
        optim = [torch.optim.SGD(parlist[k], lr=args["lr"]) for k in range(N)]
    else:
        raise Exception("Optimizer %s is not implemented"%args["optim"])
    
    minloss = 1000000
    noprogress = 0
    for n in range(args["epochs"]):
        time_start = time.time()
        tot_loss = 0
        for i in range(N):
            pred = Ug[i]@Vg[i].T+ Ul[i]@Vl[i].T
            lossi = torch.sum((Ymask[i]*(Y[i]-pred))**2)
            optim[i].zero_grad()
            lossi.backward()
            optim[i].step()
            
            tot_loss += lossi.item()
        with torch.no_grad():
            Ug_avg *= 0
            Ug_avg += sum([Ug[i] for i in range(N)])/N
            pj0 = torch.inverse(Ug_avg.T@Ug_avg)@Ug_avg.T
            
            projection = Ug_avg@pj0
            for i in range(N):
                Ug[i] *= 0
                Ug[i] += Ug_avg
                Vg[i] += Vl[i]@Ul[i].T@pj0.T
                Ul[i] -= projection@Ul[i]
      

        tot_loss /= N

        # if the loss is too small, break
        if "break_on_epsilon" in args.keys():
            if tot_loss < args["break_on_epsilon"]:
                break

        # if there is no progress for many iterations, break
        
        if tot_loss < minloss:
            minloss = tot_loss
            noprogress = 0
        else:
            noprogress += 1
        if 'noprogressthreshold' in args.keys():
            if noprogress > args['noprogressthreshold']:
                print("reduceing the stepsize, loss %.6f"%tot_loss)
                with torch.no_grad():
                    for k in range(N):
                        for i, param_group in enumerate(optim[k].param_groups):
                            param_group['lr']*= np.exp(-1)
                print(param_group['lr'])
                noprogress = 0
        
        output = "[%s/%s], loss %.6f, "%(n,args["epochs"],tot_loss)
        if "global_subspace_err_metric" in args.keys():
            output += " gserr %s "%args["global_subspace_err_metric"](Ug_avg)
        if "local_subspace_err_metric" in args.keys():
            output += " lserr %s "%args["local_subspace_err_metric"](Ul)
            output += " adderr %s "% (args["global_subspace_err_metric"](Ug_avg)+args["local_subspace_err_metric"](Ul))

        if "global_recovery_error" in args.keys():
            output += " g_recovery_err %.8f, "%args["global_recovery_error"]([Ug,Vg,Ul,Vl])
        if "local_recovery_error" in args.keys():
            output += " l_recovery_err %.8f, "%args["local_recovery_error"]([Ug,Vg,Ul,Vl])
        
        time_end = time.time()
        if (args['verbose']>0.5 and n%(args["epochs"]//10+1)==0) or args['verbose'] >10 or n == args["epochs"]-1:
            print(output+", time %s"%(time_end-time_start))
        #print("[%s/%s], loss %s"%(n,args["epochs"],tot_loss))
    
    return Ug, Vg, Ul, Vl



