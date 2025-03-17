import torch
import numpy as np
import os

class Poisson_equation(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def f(self,X):
        x,y = X[:,0],X[:,1]
        result = torch.exp(-x)*(x-2+y**3+6*y)
        return result
    def strong_form(self,X,grad_result):
        # du_dx,du_dy,d2u_dx2,d2u_dy2 = grad_result["du_dx"],grad_result["du_dy"],grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        d2u_dx2,d2u_dy2 = grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        result = d2u_dx2 + d2u_dy2 - self.f(X)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 1/2*(du_dx**2 + du_dy**2) + self.f(X)*u
        return result
    def BC_function(self,X):
        result = torch.exp(-X[:,0])*(X[:,0]+X[:,1]**3)
        return result
    def real_solution(self,X):
        result = torch.exp(-X[:,0])*(X[:,0]+X[:,1]**3)
        return result


class multiscale_equation(torch.nn.Module):
    def __init__(self,eps = 0.125):
        super().__init__()
        self.eps = eps
        self.result_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "Multiscale","true")
    def f(self,X):
        x,y = X[:,0],X[:,1]
        result = torch.sin(x)+torch.cos(y)
        return result
    def a(self,X):
        x,y = X[:,0],X[:,1]
        result = 2 + torch.sin(2*torch.pi*x/self.eps)*torch.cos(2*torch.pi*y/self.eps)
        return result
    def da_dx(self,X):
        x = X[:,0]
        result = 2*torch.pi/self.eps*torch.cos(2*torch.pi*x/self.eps)
        return result
    def da_dy(self,X):
        y = X[:,1]
        result = -2*torch.pi/self.eps*torch.sin(2*torch.pi*y/self.eps)
        return result
    def strong_form(self,X,grad_result):
        du_dx,du_dy,d2u_dx2,d2u_dy2 = grad_result["du_dx"],grad_result["du_dy"],grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        # da/dx*du/dx + da/dy*du/dy + a*d2u/dx2 + a*d2u/dy2 + f = 0
        result = self.da_dx(X)*du_dx + self.da_dy(X)*du_dy + self.a(X)*(d2u_dx2 + d2u_dy2) + self.f(X)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 1/2*self.a(X)*(du_dx**2 + du_dy**2) - self.f(X)*u
        return result
    def BC_function(self,X):
        result = X[:,0]*0
        return result
    def real_solution(self,X):
        file_path = os.path.join(self.result_path,"u_eval_eps={}.txt".format(self.eps))
        result = torch.tensor(np.loadtxt(file_path))
        return result

class microscale_Poisson_PBC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.result_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "microscale_pbc","true")
    def a(self,X):
        x,y = X[:,0],X[:,1]
        result = 2 + torch.sin(2*torch.pi*x)*torch.cos(2*torch.pi*y)
        return result
    def f(self,X):
        x,y = X[:,0],X[:,1]
        result = 2*torch.pi*torch.cos(2*torch.pi*x)*torch.cos(2*torch.pi*y)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 1/2*self.a(X)*(du_dx**2 + du_dy**2) - self.f(X)*u
        return result
    def real_solution(self,X):
        file_path = os.path.join(self.result_path,"u_eval.txt")
        result = torch.tensor(np.loadtxt(file_path))
        return result

class Phase_field_equation_1d(torch.nn.Module):
    def __init__(self,l = 0.025):
        super().__init__()
        self.l = l
        self.l_rcpl = 1/l
    def f(self,X):
        return 0
    def strong_form(self,X,grad_result):
        u = grad_result["u"]
        d2u_dx2 = grad_result["d2u_dx2"]
        d2u_dy2 = grad_result["d2u_dy2"]
        result = u - self.l**2*(d2u_dx2 + d2u_dy2)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 1/2*(u**2 + self.l**2*(du_dx**2 + du_dy**2))
        return result
    def BC_function(self,X):
        result = torch.exp(-torch.abs(X[:,0])*self.l_rcpl)
        return result
    def real_solution(self,X):
        result = torch.exp(-torch.abs(X[:,0])*self.l_rcpl)
        return result
    
class High_frequency_Poisson_equation(torch.nn.Module):
    def __init__(self,frequency = 3):
        super().__init__()
        self.omeaga = 2*frequency*torch.pi
        self.a = 1/(2*self.omeaga**2)
    def f(self,X):
        x,y = X[:,0],X[:,1]
        return -torch.sin(self.omeaga*x)*torch.sin(self.omeaga*y)
    def strong_form(self,X,grad_result):
        # du_dx,du_dy,d2u_dx2,d2u_dy2 = grad_result["du_dx"],grad_result["du_dy"],grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        d2u_dx2,d2u_dy2 = grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        result = d2u_dx2 + d2u_dy2 - self.f(X)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 1/2*(du_dx**2 + du_dy**2) + self.f(X)*u
        return result
    def BC_function(self,X):
        return torch.sin(X[:,0]*torch.pi)*torch.sin(X[:,1]*torch.pi)
        # return self.real_solution(X)
    def real_solution(self,X):
        return -self.a*self.f(X)

class Helmholtz_equation(torch.nn.Module):
    def __init__(self,lam = 1, a1 = 4, a2 = 16):
        super().__init__()
        self.lam = lam
        self.a1 = a1
        self.a2 = a2
    def f(self,X):
        x,y = X[:,0],X[:,1]
        return -((self.a1*torch.pi)**2 + (self.a2*torch.pi)**2 - self.lam) \
                *torch.sin(self.a1*torch.pi*x)*torch.sin(self.a2*torch.pi*y)
        
    def strong_form(self,X,grad_result):
        u = grad_result["u"]
        d2u_dx2,d2u_dy2 = grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        result = d2u_dx2 + d2u_dy2 + self.lam*u - self.f(X)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 0.5*(du_dx**2 + du_dy**2) - 0.5*self.lam*u**2 + self.f(X)*u
        return result
    def BC_function(self,X):
        # return torch.sin(torch.pi*X[:,0])*torch.sin(torch.pi*X[:,1])
        return X[:,0]*(1-X[:,0])*X[:,1]*(1-X[:,1])
        # return self.real_solution(X)
    def real_solution(self,X):
        return torch.sin(self.a1*torch.pi*X[:,0])*torch.sin(self.a2*torch.pi*X[:,1])


'''
class Heat_equation(torch.nn.Module):
    def __init__(self,k = 1, L = 1):
        super().__init__()
        self.k = k
        self.L = L
        self.tau = 1e-4
    def strong_form(self,grad_result):
        du_dt,d2u_dx2 = grad_result["du_dx"],grad_result["d2u_dy2"]
        result = du_dt - self.k*d2u_dx2
        return result
    def variational_energy(self,X,u,du_dt,du_dx):
        # result = 1/2*(du_dt**2 +self.k* du_dx**2)
        result = 0.5*(self.tau*du_dt**2 + self.k* du_dx**2)*torch.exp(-X[:,0]/self.tau)
        return result
    def BC_function(self,X):
        # return torch.sin(X[:,0]*torch.pi)*torch.sin(X[:,1]*torch.pi)
        return self.real_solution(X)
    def real_solution(self,X):
        return torch.exp(-self.k*(torch.pi/self.L)**2*X[:,0])*torch.sin(X[:,1]*torch.pi/self.L)

'''
class Wave_equation(torch.nn.Module):
    def __init__(self,c = 1.0, L = 1):
        super().__init__()
        self.c = c
        self.c2 = c**2
        self.L = L
    def strong_form(self,grad_result):
        d2u_dt2,d2u_dx2 = grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        result = d2u_dt2 - self.c2*d2u_dx2
        return result
    def variational_energy(self,X,u,du_dt,du_dx):
        # result = 1/2*(du_dt**2 +self.k* du_dx**2)
        result = 0.5*(du_dt**2 - self.c2* du_dx**2)
        return result
    def BC_function(self,X):
        # return torch.sin(X[:,0]*torch.pi)*torch.sin(X[:,1]*torch.pi)
        return self.real_solution(X)
    def real_solution(self,X):
        # return torch.sin(X[:,1]*torch.pi/self.L)*torch.cos(self.c*X[:,0]*torch.pi/self.L)
        return torch.sin(X[:,1] + self.c*X[:,0])
    

class High_contrast_Poisson(torch.nn.Module):
    def __init__(self,eps = 1.0, ratio = 1e6, a_min = 1):
        super().__init__()
        self.eps = eps
        self.ratio = ratio
        self.a_min = a_min
        self.factor_1 = (ratio-1)*a_min/2.0
        self.factor_2 = self.factor_1 + a_min
        self.result_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "High_contrast_Poisson","true")
    def f(self,X):
        x,y = X[:,0],X[:,1]
        result = (torch.sin(x)+torch.cos(y))*self.ratio
        return result
    def a(self,X):
        x,y = X[:,0],X[:,1]
        result = self.factor_2 + torch.sin(2*torch.pi*x/self.eps)*torch.cos(2*torch.pi*y/self.eps)*self.factor_1
        return result
    def da_dx(self,X):
        x = X[:,0]
        result = 2*torch.pi/self.eps*torch.cos(2*torch.pi*x/self.eps)*self.factor_1
        return result
    def da_dy(self,X):
        y = X[:,1]
        result = -2*torch.pi/self.eps*torch.sin(2*torch.pi*y/self.eps)*self.factor_1
        return result
    def strong_form(self,X,grad_result):
        du_dx,du_dy,d2u_dx2,d2u_dy2 = grad_result["du_dx"],grad_result["du_dy"],grad_result["d2u_dx2"],grad_result["d2u_dy2"]
        # da/dx*du/dx + da/dy*du/dy + a*d2u/dx2 + a*d2u/dy2 + f = 0
        result = self.da_dx(X)*du_dx + self.da_dy(X)*du_dy + self.a(X)*(d2u_dx2 + d2u_dy2) + self.f(X)
        return result
    def variational_energy(self,X,u,du_dx,du_dy):
        result = 1/2*self.a(X)*(du_dx**2 + du_dy**2) - self.f(X)*u
        return result
    def BC_function(self,X):
        result = X[:,0]*0
        return result
    def real_solution(self,X):
        file_path = os.path.join(self.result_path,"u_eval_eps={}_ratio={}.txt".format(self.eps,self.ratio))
        result = torch.tensor(np.loadtxt(file_path))
        return result