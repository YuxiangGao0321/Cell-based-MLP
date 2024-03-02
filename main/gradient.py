import torch
from torch.autograd import grad

class grad1(torch.nn.Module):
    # def __init__(self,uv,output_hessian = True,vectorize = False):
    def __init__(self,f, batch_size):
        super().__init__()
        self.f = f
        # self.grad_ones = torch.ones_like(f(X_for_shape.to("cuda")),device="cuda")
        self.grad_ones = torch.ones(batch_size,device="cuda")
        # self.output_hessian = output_hessian
        # self.vectorize = vectorize

    def forward_2d(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        u = self.f(X)

        grad_u = grad(u,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        # grad_u = grad(u,X,self.grad_ones)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]

        return u_x,u_y,u

    def forward_2d_plot(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        grad_ones = torch.ones(X.shape[0],device="cuda")
        u = self.f(X)

        grad_u = grad(u,X,grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]

        return u_x,u_y


    def forward_2d_uv(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        u = self.f(X)

        grad_u = grad(u[:,0],X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]
        grad_v = grad(u[:,1],X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        v_x,v_y = grad_v[:,0],grad_v[:,1]

        return u_x,u_y,v_x,v_y

    def forward_2d_uv_plot(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        grad_ones = torch.ones(X.shape[0],device="cuda")
        u = self.f(X)

        grad_u = grad(u[:,0],X,grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]
        grad_v = grad(u[:,1],X,grad_ones,retain_graph=True, create_graph=True)[0]
        v_x,v_y = grad_v[:,0],grad_v[:,1]

        return u_x,u_y,v_x,v_y


    def forward_2d_uv_2out(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        ux,uy = self.f(X)

        grad_u = grad(ux,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]
        grad_v = grad(uy,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        v_x,v_y = grad_v[:,0],grad_v[:,1]

        return u_x,u_y,v_x,v_y

    def forward_2d_uv_2out_plot(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        grad_ones = torch.ones(X.shape[0],device="cuda")
        ux,uy = self.f(X)

        grad_u = grad(ux,X,grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]
        grad_v = grad(uy,X,grad_ones,retain_graph=True, create_graph=True)[0]
        v_x,v_y = grad_v[:,0],grad_v[:,1]

        return u_x,u_y,v_x,v_y
   
    def forward_3d(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        u = self.f(X)

        grad_u = grad(u,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y,u_z = grad_u[:,0],grad_u[:,1],grad_u[:,2]

        return u_x,u_y,u_z

    def to_device(self,device):
        self.grad_ones = self.grad_ones.to(device)


class grad2(torch.nn.Module):
    def __init__(self,f,batch_size):
        super().__init__()
        self.f = f
        self.grad_ones = torch.ones(batch_size)

    def to_device(self,device):
        self.grad_ones = self.grad_ones.to(device)

    def forward(self,X):
        X.requires_grad = True    
        u = self.f(X)

        grad_u = grad(u,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]
        grad_dudx = grad(u_x,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        grad_dudy = grad(u_y,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_xx,u_xy,u_yy = grad_dudx[:,0],grad_dudx[:,1],grad_dudy[:,1]
        # Laplace_f = u_xx + u_yy


        res = {
            'u':u,
            'du_dx':u_x,
            'du_dy':u_y,
            'd2u_dx2':u_xx,
            'd2u_dxy':u_xy,
            'd2u_dy2':u_yy,
            # 'Laplace_f':Laplace_f
        }

        return res