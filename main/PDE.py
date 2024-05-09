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