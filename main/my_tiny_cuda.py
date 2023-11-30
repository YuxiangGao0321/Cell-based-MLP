import torch.nn as nn
import torch
import numpy as np
from torch.autograd import grad
import os


def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def my_rmse(X1,X2):
    return np.sqrt(((X1-X2)**2).mean())

def my_relativeL2(pred,true):
    error = pred-true
    return np.sqrt((error**2).sum()/(true**2).sum())

class my_encoding_linear(nn.Module):
    def __init__(self, n_input_dims=2, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=20, base_resolution=16, per_level_scale=1.26):
        super().__init__()
        # configurations
        self.n_input_dims = n_input_dims
        self.L = n_levels
        self.F = n_features_per_level
        self.T = 2**log2_hashmap_size
        self.N_min = base_resolution
        self.bb = per_level_scale


        # hash table
        self.hash_table = torch.nn.Parameter(torch.rand(self.T,self.F,self.L)*(2e-4)-(1e-4),requires_grad=True)
        
        # constants
        self.NL = torch.floor(self.N_min*self.bb**torch.arange(self.L)).unsqueeze(0).unsqueeze(0)
        self.pi1 = 1
        self.pi2 = 2654435761      

    def linear_h(self, x0):

        # x.shape = (batch_size,n_input_dims,n_levels)
        # x is the coordinate of the left bottom corner
        # 2D
        # h2 -- h3
        # |      |
        # h0 -- h1

        h0 = torch.bitwise_xor(x0[:,0,:]*self.pi1,x0[:,1,:]*self.pi2)%self.T
        h1 = torch.bitwise_xor((x0[:,0,:]+1)*self.pi1,x0[:,1,:]*self.pi2)%self.T
        h2 = torch.bitwise_xor(x0[:,0,:]*self.pi1,(x0[:,1,:]+1)*self.pi2)%self.T
        h3 = torch.bitwise_xor((x0[:,0,:]+1)*self.pi1,(x0[:,1,:]+1)*self.pi2)%self.T

        return h0, h1, h2, h3


    def bilinear_interpolate(self, local_coords, h0, h1, h2, h3):

        # h0.shape = (batch_size, n_levels)
        # h0 to h3 are the key values of hash table

        # local_coords.shape = (batch_size, n_input_dims, n_levels)

        n_batch = h0.shape[0]

        level_select = (torch.arange(self.L).unsqueeze(0)*torch.ones((n_batch,1))).to(torch.int64).to(local_coords.device)

        H0 = self.hash_table[h0,:,level_select]
        H1 = self.hash_table[h1,:,level_select]
        H2 = self.hash_table[h2,:,level_select]
        H3 = self.hash_table[h3,:,level_select]


        x1, x2 = local_coords[:,0,:].unsqueeze(self.n_input_dims),local_coords[:,1,:].unsqueeze(self.n_input_dims)


        y = ((1-x1)*(1-x2)*H0 + x1*(1-x2)*H1 + (1-x1)*x2*H2 + x1*x2*H3).view(n_batch,-1)

        # y.shape = (batch_size, n_features_per_level*n_levels)

        return y

    def forward(self, x):

        # x.shape = (batch_size, n_input_dims)
        # x = [ [x11, x12], [x21, x22], ..., [xn1, xn2] ]

        x = x.squeeze().unsqueeze(-1)
        
        xNL = x * (self.NL).to(x.device)
        # xNL_floor = torch.floor(xNL).to(torch.int64)
        xNL_floor = xNL.to(torch.int64) # xNL must >= 0
        h0, h1, h2, h3 = self.linear_h(xNL_floor)
        local_coords = xNL - xNL_floor
        y = self.bilinear_interpolate(local_coords, h0, h1, h2, h3)
        return y

class my_grid_linear(nn.Module):
    def __init__(self, n_input_dims=2, n_levels=16, n_features_per_level=2,
        base_resolution=16, per_level_scale=1.26, 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),grad_fix = False):
        super().__init__()
        # configurations
        self.n_input_dims = n_input_dims
        self.L = n_levels
        self.F = n_features_per_level
        self.N_min = base_resolution
        self.bb = per_level_scale
        self.N_max = int(self.N_min * self.bb**(self.L-1))
        self.feature_vector_size = self.L*self.F
        if grad_fix:
            from torch_utils.ops import grid_sample_gradfix
            self.sampler = grid_sample_gradfix.grid_sample
        else:
            self.sampler = torch.grid_sample



        # hash table
        self.grid_table = torch.nn.Parameter(torch.rand(self.L,self.F,self.N_max,self.N_max)*(2e-4)-(1e-4)
            ,requires_grad=True).to(device)
        
        # constants
        self.scaler = ((self.N_min*self.bb**torch.arange(self.L)).to(torch.int64)/512).unsqueeze(-1).unsqueeze(-1).to(device)

    def forward(self, x):

        # x.shape = (batch_size, n_input_dims)
        # x = [ [x11, x12], [x21, x22], ..., [xn1, xn2] ]

        x = x.unsqueeze(0)

        x_local = x * self.scaler
        '''
        # x_local.shape = (n_levels, batch_size, n_input_dims)
        # self.grid_table.shape = (n_levels, n_features_per_level, max_resolution, max_resolution)
        # y.shape = (n_levels, n_features_per_level, batch_size)
        '''
        # y = torch.nn.functional.grid_sample(self.grid_table, x_local.unsqueeze(1),align_corners=False).squeeze()
        y = self.sampler(self.grid_table, x_local.unsqueeze(1)).squeeze() #align_corners=False
        # y = grid_sample_gradfix._GridSample2dForward.apply(self.grid_table, x_local.unsqueeze(1)).squeeze()
        
        y = torch.transpose(y, 0, 2).reshape(-1,self.feature_vector_size)
        return y

class my_encoding_cubic(nn.Module):
    def __init__(self, n_input_dims=2, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=22, base_resolution=16, per_level_scale=1.26, device = "cuda"):
        super().__init__()
        # configurations
        # self.n_batch = n_batch
        self.n_input_dims = n_input_dims
        self.L = n_levels
        self.F = n_features_per_level
        self.T = 2**log2_hashmap_size
        self.N_min = base_resolution
        self.bb = per_level_scale

        self.width = int(n_levels*n_features_per_level)

        # hash table
        self.hash_table = torch.nn.Parameter(torch.rand(self.T,self.F,self.L)*(2e-4)-(1e-4),requires_grad=True).to(device)
        
        # constants
        self.NL = torch.floor(self.N_min*self.bb**torch.arange(self.L)).unsqueeze(0).unsqueeze(0).to(device)
        self.pi1 = 1
        self.pi2 = 2654435761

        # self.level_select = (torch.arange(self.L).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #     *torch.ones((n_batch,4,4,1))).to(torch.int64).to(device)
        self.level_select = torch.arange(self.L).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.B = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [-3, 3, -2, -1],
            [2, -2, 1, 1]
        ]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float().to(device)

        # self.local_node = torch.tensor([
        #     [[-1,-1],[0,-1],[1,-1],[2,-1]],
        #     [[-1,0],[0,0],[1,0],[2,0]],
        #     [[-1,1],[0,1],[1,1],[2,1]],
        #     [[-1,2],[0,2],[1,2],[2,2]],
        #     ]).unsqueeze(0).unsqueeze(-1).to(device)
        self.local_node = torch.tensor([
            [[-1,-1],[-1,0],[-1,1],[-1,2]],
            [[0,-1],[0,0],[0,1],[0,2]],
            [[1,-1],[1,0],[1,1],[1,2]],
            [[2,-1],[2,0],[2,1],[2,2]],
            ]).unsqueeze(0).unsqueeze(-1).to(device) + 1

        self.xxyy_power = torch.tensor([0,1,2,3]).to(device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    def cubic_h(self, x0):

        # x.shape = (batch_size,n_input_dims,n_levels)
        # x is the coordinate of point h11
        # 2D

        # (batch_size,1,1,n_input_dims,n_levels)-(1,4,4,n_input_dims,1)
        # = (batch_size,4,4,n_input_dims, n_levels)
        xij = x0.unsqueeze(1).unsqueeze(1) + self.local_node 
        # print(xij.max(),xij.min())

        hij = torch.bitwise_xor(xij[:,:,:,0,:]*self.pi1,xij[:,:,:,1,:]*self.pi2)%self.T

        # hij.shape = (batch_size,4,4,n_levels)
        return hij

    def bicubic_interpolate(self, local_coords, hij):

        # local_coords.shape = (batch_size, n_input_dims, n_levels)
        # hij.shape = (batch_size,4,4,n_levels)

        # n_batch = local_coords.shape[0]

        # level_select = (torch.arange(self.L).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #     *torch.ones((n_batch,4,4,1))).to(torch.int64).to(device)

        f = self.hash_table[hij,:,self.level_select] # f.shape = (batch_size,4,4,n_levels,n_features_per_level)

        fy = (f[:,:,2:,:,:] - f[:,:,:2,:,:])/2 # fx.shape = (batch_size, 4, 2, n_levels,n_features_per_level)
        fx = (f[:,2:,:,:,:] - f[:,:2,:,:,:])/2 # fy.shape = (batch_size, 2, 4, n_levels,n_features_per_level)
        # fxy.shape = (batch_size, 2, 2, n_levels,n_features_per_level)
        fxy = (fy[:,2:,:,:,:] - fy[:,:2,:,:,:] + fx[:,:,2:,:,:] - fx[:,:,:2,:,:])/4
        fi = f[:,1:3,1:3,:,:]
        fiy, fix = fy[:,1:3,:,:,:], fx[:,:,1:3,:,:]


        F = torch.cat((torch.cat((fi,fiy),2),torch.cat((fix,fxy),2)),1)



        # xxyy = torch.ones([n_batch,2,4,self.L]).to(device)

        # for i in range(3):
        #     self.xxyy[:,:,i,:] = local_coords**(3-i)
        # xxyy = local_coords.unsqueeze(2)**self.xxyy_power
        xxyy = (local_coords.unsqueeze(2)**self.xxyy_power).unsqueeze(1).unsqueeze(-1)
        # print(local_coords.unsqueeze(2).shape,self.xxyy_power.shape,xxyy.shape)

        # xx = xxyy[:,0,:,:].unsqueeze(1).unsqueeze(-1) #xx.shape = (batch_size,1,4,n_levels,1)
        # yy = xxyy[:,1,:,:].unsqueeze(2).unsqueeze(-1) #yy.shape = (batch_size,4,1,n_levels,1)
        # print(Q.shape,self.xxyy_power.shape,xxyy.shape)
        #self.B.shape = (1,4,4,1,1)
        # print(self.B.shape,Q.shape)

        result = torch.einsum("abcde,fcghi->fbghi",self.B,F)
        result = torch.einsum("abcde,fgchi->abgde",result,self.B)
        result = torch.einsum("abcde,acfdg->abfdg",xxyy[:,:,0,:,:],result)
        result = torch.einsum("abcde,afcdg->abfde",result,xxyy[:,:,1,:,:])
        # print(result.shape)
        result = (result).squeeze().view(-1,self.width)
        # print(result.shape)

        return result

    def forward(self, x):

        # x.shape = (batch_size, n_input_dims)
        # x = [ [x11, x12], [x21, x22], ..., [xn1, xn2] ]

        x = x.squeeze().unsqueeze(-1)
        # print(x[1],self.NL)
        
        xNL = x * self.NL
        
        xNL_floor = xNL.to(torch.int64) # xNL must >= 0
        # print(xNL[1],xNL_floor[1])
        hij = self.cubic_h(xNL_floor)
        local_coords = xNL - xNL_floor
        y = self.bicubic_interpolate(local_coords, hij)
        return y


class my_MLP(nn.Module):
    def __init__(self, n_input_dims = None, n_output_dims=1, 
        n_levels=16, n_features_per_level=2, n_hidden = 2, width = 64,
        activation = nn.ReLU(),spectral_norm = False,dtype = torch.float32):
        super().__init__()

        if n_input_dims == None:
            in_dim = n_levels * n_features_per_level
        else:
            in_dim = n_input_dims
        
        dim_list = np.ones((n_hidden+2)) * width
        dim_list[0] = in_dim
        dim_list[-1] = n_output_dims
        dim_list = dim_list.astype("int")
        # print(dim_list)
        # dim_list = [in_dim, 64, 64,64, n_output_dims]

        blocks = []
        if spectral_norm:
            for i in range(len(dim_list)-2):
                blocks.append(nn.utils.parametrizations.spectral_norm(nn.Linear(dim_list[i],dim_list[i+1],dtype = dtype)))
                blocks.append(activation)
        else:            
            for i in range(len(dim_list)-2):
                blocks.append(nn.Linear(dim_list[i],dim_list[i+1],dtype = dtype))
                blocks.append(activation)

        blocks.append(nn.Linear(dim_list[-2],dim_list[-1],dtype = dtype))

        self.my_seq = nn.Sequential(*blocks)
    def forward(self, x):
        return self.my_seq(x).squeeze()

class my_MLP2(nn.Module):
    def __init__(self, n_input_dims = None, n_output_dims=1, 
        n_levels=16, n_features_per_level=2, n_hidden = 2, width = 64,
        activation = nn.ReLU(),spectral_norm = False,dtype = torch.float32):
        super().__init__()

        if n_input_dims == None:
            in_dim = n_levels * n_features_per_level
        else:
            in_dim = n_input_dims
        
        dim_list = np.ones((n_hidden+2)) * width
        dim_list[0] = in_dim
        dim_list[-1] = n_output_dims
        dim_list = dim_list.astype("int")
        # print(dim_list)
        # dim_list = [in_dim, 64, 64,64, n_output_dims]

        blocks = []
        if spectral_norm:
            for i in range(len(dim_list)-2):
                blocks.append(nn.utils.parametrizations.spectral_norm(nn.Linear(dim_list[i],dim_list[i+1],dtype = dtype)))
                blocks.append(activation)
        else:            
            for i in range(len(dim_list)-2):
                blocks.append(nn.Linear(dim_list[i],dim_list[i+1],dtype = dtype))
                blocks.append(activation)

        blocks.append(nn.utils.parametrizations.spectral_norm(nn.Linear(dim_list[-2],dim_list[-1],dtype = dtype)))

        self.my_seq = nn.Sequential(*blocks)
    def forward(self, x):
        return self.my_seq(x).squeeze()



class my_sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.sin(x)



def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        c = np.sqrt(6/m.in_features)
        m.weight.data.uniform_(-c, c)
        m.bias.data.fill_(0)


class my_FF(nn.Module):
    def __init__(self, n_input_dims = 1,sigma = 20,width = 64,dtype = torch.float32):
        super().__init__()
        # self.B = torch.randn((int(width/2),n_input_dims))*sigma # B.shape = (width/2, n_dim) in N(0,sigma)
        self.B = torch.randn((n_input_dims,int(width/2)), dtype=dtype)*sigma*2*torch.pi # B.shape = (n_dim,width/2) in N(0,sigma)
    def forward(self,x):
        # input.shape = (n_batch,n_dim)
        x = torch.matmul(x,self.B.to(x.device))
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class my_FF_multi(nn.Module):
    def __init__(self, n_input_dims = 1,sigma_list = [1,5,10,50],width = 64,dtype = torch.float32):
        super().__init__()
        self.n_FF = len(sigma_list)
        self.width_half = int(width/2)
        self.sigma_list = torch.tensor(sigma_list).unsqueeze(0).unsqueeze(0)*2*torch.pi
        self.B0 = torch.randn((n_input_dims,int(self.width_half/self.n_FF),self.n_FF), dtype=dtype) 
        self.B = self.B0*self.sigma_list # B.shape = (n_dim,width/2/n_FF, n_FF)
    def forward(self,x):
        # input.shape = (n_batch,n_dim) --> (n_batch,n_dim,1)
        xB = torch.einsum('abc,bde->ade', x.unsqueeze(-1), self.B.to(x.device)).view(-1,self.width_half)
        return torch.cat([torch.sin(xB), torch.cos(xB)], dim=1)

class my_FFNN(nn.Module):
    def __init__(self, n_input_dims = 1, n_output_dims=1, sigma = 20,
        n_hidden = 2, width = 64,activation = nn.ReLU(),dtype = torch.float32):
        super().__init__()
        Fourier_embedding = my_FF(n_input_dims =n_input_dims,sigma = sigma,width = width,dtype = dtype)
        MLP = my_MLP(n_input_dims = width, n_output_dims=n_output_dims, n_hidden = n_hidden, 
            width = width,activation =activation,dtype = dtype)

        self.my_seq = nn.Sequential(Fourier_embedding,MLP)
    def forward(self, x):
        return self.my_seq(x)

class my_FFNN_multi(nn.Module):
    def __init__(self, n_input_dims = 1, n_output_dims=1, sigma_list = [1,5,10,50],
        n_hidden = 2, width = 64,activation = nn.ReLU(),dtype = torch.float32):
        super().__init__()
        Fourier_embedding = my_FF_multi(n_input_dims =n_input_dims,sigma_list = sigma_list,width = width,dtype = dtype)
        MLP = my_MLP(n_input_dims = width, n_output_dims=n_output_dims, n_hidden = n_hidden, 
            width = width,activation =activation,dtype = dtype)
        self.my_seq = nn.Sequential(Fourier_embedding,MLP)

    def forward(self, x):
        return self.my_seq(x)



class fdiff1(torch.nn.Module):
    # def __init__(self,uv,output_hessian = True,vectorize = False):
    def __init__(self,f, X_for_shape):
        super().__init__()
        self.f = f
        self.grad_ones = torch.ones_like(f(X_for_shape.to("cuda")),device="cuda")
        # self.output_hessian = output_hessian
        # self.vectorize = vectorize
    
    def forward(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        u = self.f(X)

        grad_u = grad(u,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y = grad_u[:,0],grad_u[:,1]

        return u_x,u_y

class fdiff2(torch.nn.Module):
    # def __init__(self,uv,output_hessian = True,vectorize = False):
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
            'u_x':u_x,
            'u_y':u_y,
            'u_xx':u_xx,
            'u_xy':u_xy,
            'u_yy':u_yy,
            # 'Laplace_f':Laplace_f
        }

        return res


class fdiff1_3d(torch.nn.Module):
    # def __init__(self,uv,output_hessian = True,vectorize = False):
    def __init__(self,f, X_for_shape):
        super().__init__()
        self.f = f
        self.grad_ones = torch.ones_like(f(X_for_shape.to("cuda")),device="cuda")
        # self.output_hessian = output_hessian
        # self.vectorize = vectorize
    
    def forward(self,X):
        # X = X_original.clone()
        X.requires_grad = True
        # print(X.dtype)
        u = self.f(X)

        grad_u = grad(u,X,self.grad_ones,retain_graph=True, create_graph=True)[0]
        u_x,u_y,u_z = grad_u[:,0],grad_u[:,1],grad_u[:,2]

        return u_x,u_y,u_z

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

class quad_element:
    def __init__(self,x1,x2,x3,x4,n_quad = 5):
        # xi.shape = (n_elemetns, 2)
        self.X1 = np.array(x1)
        self.X2 = np.array(x2)
        self.X3 = np.array(x3)
        self.X4 = np.array(x4)
        
        gp_1d, gw_1d = np.polynomial.legendre.leggauss(n_quad)
        gpx_ref,gpy_ref = np.meshgrid(gp_1d, gp_1d)
        
        self.gp_real,self.gp_detJ = self.ref_to_real_coord(gpx_ref.reshape(-1),gpy_ref.reshape(-1))
        
        gwx, gwy = np.meshgrid(gw_1d, gw_1d)
        gw_2d = gwx.reshape(-1)*gwy.reshape(1,-1)
        
        self.gp_coords = self.gp_real.reshape(-1,2)
        self.gp_weights = (gw_2d*self.gp_detJ).reshape(-1)
        
        
    def ref_to_real_coord(self,zeta,eta):
        # zeta.shape = (n_points)
        
        zeta = np.expand_dims(zeta,axis = 0)
        eta = np.expand_dims(eta,axis = 0)
        Psi1 = (1-zeta)*(1-eta)/4
        Psi2 = (1+zeta)*(1-eta)/4
        Psi3 = (1+zeta)*(1+eta)/4
        Psi4 = (1-zeta)*(1+eta)/4
        
        # real_coords.shape = (n_elements,n_points,2)
        real_coords = np.array([Psi1*self.X1[:,0:1]+Psi2*self.X2[:,0:1]+Psi3*self.X3[:,0:1]+Psi4*self.X4[:,0:1],
                Psi1*self.X1[:,1:2]+Psi2*self.X2[:,1:2]+Psi3*self.X3[:,1:2]+Psi4*self.X4[:,1:2]]).transpose(1,2,0)
        #dref.shape = (1,n_points,2,4)
        dref = np.array([[-(1-eta), 1-eta, 1+eta, -(1+eta)],[-(1-zeta),-(1+zeta),1+zeta,1-zeta]]).transpose(2,3,0,1)
        #real_nodes.shape = (n_elements,1,4,2)
        real_nodes = np.array([[self.X1,self.X2,self.X3,self.X4]],
            dtype = np.float32).transpose(2,0,1,3)
        #J.shape = (n_elements,n_points,2,2)
        J = np.einsum('abcd,efdh->ebch',dref,real_nodes)/4
        #detJ.shape = (n_elements,n_points)
        detJ = np.abs(np.linalg.det(J))
        return real_coords,detJ



def plot_diff(x,ground_truth,field_name,f_name,eps = 1e-4,ifsave = False,ifplot = True,
    L = 1):
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from matplotlib import colors
    plt.rcParams.update({'font.size': 15})
    resolution = x.shape[0]
    xmin,ymin = 0,0
    xmax,ymax = L,L
    fig = plt.figure(figsize=(15,4))

    plt.subplot(131)
    # fmax = float(np.format_float_positional(x.max(),precision = 3,unique=False, fractional=False, trim='k'))
    # fmin = float(np.format_float_positional(x.min(),precision = 3,unique=False, fractional=False, trim='k'))
    fmax = float(np.format_float_positional(ground_truth.max(),precision = 3,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(ground_truth.min(),precision = 3,unique=False, fractional=False, trim='k'))

    interval = (fmax-fmin)/4
    ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.imshow(x,origin = 'lower',interpolation = None,
    vmin=fmin,vmax=fmax)#,cmap = cm.binary)
    # plt.imshow(f_plot,origin = 'lower',interpolation = None)
    plt.xticks([0,resolution],[xmin,xmax])
    plt.yticks([0,resolution],[ymin,ymax])
    plt.title(field_name)
    plt.colorbar(ticks=ct)
    # plt.axis('off')

    plt.subplot(132)
    # fmax = float(np.format_float_positional(ground_truth.max(),precision = 3,unique=False, fractional=False, trim='k'))
    # fmin = float(np.format_float_positional(ground_truth.min(),precision = 3,unique=False, fractional=False, trim='k'))
    # interval = (fmax-fmin)/4
    # ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.imshow(ground_truth,origin = 'lower',interpolation = None,
    vmin=fmin,vmax=fmax)#,cmap = cm.binary)
    plt.xticks([0,resolution],[xmin,xmax])
    plt.yticks([0,resolution],[ymin,ymax])
    plt.title("ground_truth")
    plt.colorbar(ticks=ct)
    # plt.axis('off')

    
    plt.subplot(133)
    diff = x - ground_truth
    # rmse = float(np.format_float_positional(my_rmse(x, ground_truth),precision = 4,unique=False, fractional=False, trim='k'))
    nrmse = float(np.format_float_positional(my_relativeL2(x, ground_truth),precision = 4,
        unique=False, fractional=False, trim='k'))
    fmax = float(np.format_float_positional(diff.max(),precision = 3,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(diff.min(),precision = 3,unique=False, fractional=False, trim='k'))
    if fmin<-eps and fmax>eps:
        ct = (fmin, 0.5*fmin, 0, 0.5*fmax, fmax)
    else:
        interval = (fmax-fmin)/4
        ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.imshow(diff,origin = 'lower',norm=colors.TwoSlopeNorm(vcenter=ct[2],vmin = ct[0],vmax = ct[-1]),
              interpolation = None,cmap = cm.RdBu_r)
    # plt.imshow(f_plot,origin = 'lower',interpolation = None)
    plt.xticks([0,resolution],[xmin,xmax])
    plt.yticks([0,resolution],[ymin,ymax])
    plt.title("{} - real(NRMSE: {})".format(field_name,nrmse))
    plt.colorbar(ticks=ct)
    plt.axis('off')
    fig.tight_layout()
    if ifsave:
        plt.savefig('figs/{}.jpg'.format(f_name),dpi = 300)
        plt.close()
    if ifplot:
        plt.show()

def plot_result_ele(f,f0,x,mesh,name,f_name = None,mydpi = 500, L = 1,ifsave = False,ifplot = True):
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from matplotlib import colors
    eps = 1e-5
    plt.rcParams.update({'font.size': 15})
    x1,x2 = x[0],x[1]
    xmin,ymin = 0,0
    xmax,ymax = L,L
    fig = plt.figure(figsize=(15,4))
    plt.subplot(131)
    fmax = float(np.format_float_positional(f.max(),precision = 3,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(f.min(),precision = 3,unique=False, fractional=False, trim='k'))
    interval = (fmax-fmin)/4
    ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.title(name)
    plt.tripcolor(x1,x2,mesh,f,500)#,vmin=fmin,vmax=fmax)
    plt.xticks([xmin,xmax],[xmin,xmax])
    plt.yticks([ymin,ymax],[ymin,ymax])

    plt.colorbar(ticks=ct)
    plt.axis('off')
    
    plt.subplot(132)
    plt.title("Ground truth")
    plt.tripcolor(x1,x2,mesh,f0,500)#,vmin=fmin,vmax=fmax)
    plt.xticks([xmin,xmax],[xmin,xmax])
    plt.yticks([ymin,ymax],[ymin,ymax])

    plt.colorbar(ticks=ct)
    plt.axis('off')

    plt.subplot(133)
    diff = f - f0
    # rmse = float(np.format_float_positional(my_rmse(f,f0),precision = 4,unique=False, fractional=False, trim='k'))
    nrmse = float(np.format_float_positional(my_relativeL2(f, f0),precision = 4,
        unique=False, fractional=False, trim='k'))
    fmax = float(np.format_float_positional(diff.max(),precision = 3,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(diff.min(),precision = 3,unique=False, fractional=False, trim='k'))
    if fmin<-eps and fmax>eps:
        ct = (fmin, 0.5*fmin, 0, 0.5*fmax, fmax)
    else:
        interval = (fmax-fmin)/4
        ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.tripcolor(x1,x2,mesh,diff,500,
                  cmap = cm.RdBu_r,norm=colors.TwoSlopeNorm(vcenter=ct[2],vmin = ct[0],vmax = ct[-1]))
    plt.xticks([xmin,xmax],[xmin,xmax])
    plt.yticks([ymin,ymax],[ymin,ymax])

    plt.title("{} - real (NRMSE: {})".format(name,nrmse))
    plt.colorbar(ticks=ct)
    plt.axis('off')
    fig.tight_layout()
    if ifsave:
        if f_name is None:
            f_name = 'diff_{}'.format(name)
        plt.savefig('figs/{}.jpg'.format(f_name),dpi = 500)
        plt.close()
    if ifplot:
        plt.show()

def plot_result_node(f,f0,x,mesh,name,f_name = None,mydpi = 500, L = 1,ifsave = False,ifplot = True):
    from matplotlib import cm
    from matplotlib import pyplot as plt
    from matplotlib import colors
    eps = 1e-5
    plt.rcParams.update({'font.size': 15})
    x1,x2 = x[0],x[1]
    xmin,ymin = 0,0
    xmax,ymax = L,L
    fig = plt.figure(figsize=(15,4))
    plt.subplot(131)
    fmax = float(np.format_float_positional(f.max(),precision = 3,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(f.min(),precision = 3,unique=False, fractional=False, trim='k'))
    interval = (fmax-fmin)/4
    ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.title(name)
    plt.tricontourf(x1,x2,mesh,f,500)#,vmin=fmin,vmax=fmax)
    plt.xticks([xmin,xmax],[xmin,xmax])
    plt.yticks([ymin,ymax],[ymin,ymax])

    plt.colorbar(ticks=ct)
    plt.axis('off')
    
    plt.subplot(132)
    plt.title("Ground truth")
    plt.tricontourf(x1,x2,mesh,f0,500)#,vmin=fmin,vmax=fmax)
    plt.xticks([xmin,xmax],[xmin,xmax])
    plt.yticks([ymin,ymax],[ymin,ymax])

    plt.colorbar(ticks=ct)
    plt.axis('off')

    plt.subplot(133)
    diff = f - f0
    # rmse = float(np.format_float_positional(my_rmse(f,f0),precision = 4,unique=False, fractional=False, trim='k'))
    nrmse = float(np.format_float_positional(my_relativeL2(f, f0),precision = 4,
        unique=False, fractional=False, trim='k'))
    fmax = float(np.format_float_positional(diff.max(),precision = 3,unique=False, fractional=False, trim='k'))
    fmin = float(np.format_float_positional(diff.min(),precision = 3,unique=False, fractional=False, trim='k'))
    if fmin<-eps and fmax>eps:
        ct = (fmin, 0.5*fmin, 0, 0.5*fmax, fmax)
    else:
        interval = (fmax-fmin)/4
        ct = (fmin, fmin + interval, fmin + 2*interval, fmin + 3*interval, fmax)
    plt.tricontourf(x1,x2,mesh,diff,500,
                  cmap = cm.RdBu_r,norm=colors.TwoSlopeNorm(vcenter=ct[2],vmin = ct[0],vmax = ct[-1]))
    plt.xticks([xmin,xmax],[xmin,xmax])
    plt.yticks([ymin,ymax],[ymin,ymax])

    plt.title("{} - real (NRMSE: {})".format(name,nrmse))
    plt.colorbar(ticks=ct)
    plt.axis('off')
    fig.tight_layout()
    if ifsave:
        if f_name is None:
            f_name = 'diff_{}'.format(name)
        plt.savefig('figs/{}.jpg'.format(f_name),dpi = 500)
        plt.close()
    if ifplot:
        plt.show()
    



def my_sig(number,precision = 3):
    return float(np.format_float_positional(number,precision = precision,unique=False, fractional=False, trim='k'))


def hyperparameter_task_generater(file_path,parameter_range):
    import csv
    import itertools
    if not os.path.exists(file_path):
        header = ['model_idx', 'n_levels', 'base_resolution','per_level_scale', 'n_features_per_level',
        'spectral_norm','n_hidden','width',
        'lr','n_step_half_lr','n_points',
        'disp_NRMSE','strain_NRMSE','time']
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
        i = 1
    else:
        with open(file_path, "r") as file:
            i = sum(1 for line in file)
    
    data = []

    hyperparameter_list = list(itertools.product(*parameter_range))
    i0 = i
    for hyperparameter in hyperparameter_list:
        max_resolution = int(hyperparameter[1]*hyperparameter[2]**(hyperparameter[0]-1))
        if max_resolution>1000:
            continue    
        data.append([i] + list(hyperparameter))
        i = i + 1

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print('{} tasks are added in {}({} in total)'.format(i-i0,file_path,i-1))


if __name__ == "__main__":
    # print(my_MLP(n_hidden = 0))
    # import commentjson as json
    # import tinycudann as tcnn
    # my_dtype = torch.float32
    # with open("config_hash.json") as f:
    #     config = json.load(f)
    # encoding = tcnn.Encoding(2, config["encoding"],dtype = my_dtype)
    # # seed = encoding.seed
    # # params = encoding.native_tcnn_module.initial_params(seed)
    # for param in encoding.parameters():
    #     print(param)
    # print(encoding.initial_params0)
    # print(0)
    mlp=my_MLP(n_hidden=0)
    print(mlp)
    '''
    import commentjson as json
    import tinycudann as tcnn

    with open("config_hash.json") as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    my_dtype = torch.float16
    # device = torch.device("cpu")
    MAE = torch.nn.L1Loss().to(device)
    # image = Image("reference.jpg", device)

    n_batch = 3000

    

    # encoding = my_grid_linear().to(device)
    # encoding = my_encoding_linear().to(device)
    encoding = tcnn.Encoding(2, config["encoding"],dtype = my_dtype)
    # mlp = my_MLP(n_input_dims =32,activation=my_sin(),spectral_norm=False,dtype = torch.float16).to(device)
    mlp = my_MLP(n_input_dims =32,spectral_norm=False,dtype = my_dtype).to(device)
    # print(mlp)
    # model = torch.nn.Sequential(encoding,mlp)
    # model = torch.nn.Sequential(mlp)
    model = torch.nn.Sequential(encoding)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, eps=1e-15)
    diff_info = fdiff1(model)
    # diff_info = fdiff1(encoding)
    

    x = torch.rand((100,2), dtype = my_dtype).to(device)
    enc = encoding(x)
    print(encoding.loss_scale)
    print("enc:",enc.mean())

    # enocing_output = encoding(x)
    # print(enocing_output)
    # exit()

    res = diff_info(x)
    print("u_x:",res['u_x'].mean())


    weak = (res['u_x'].square()+res['u_y'].square())/2/128
    # weak = encoding(x)
    # print(weak.shape)
    loss = MAE(weak,torch.zeros_like(weak))
    # print(0)
    # print(loss.item())
    
    # optimizer.zero_grad()
    loss.backward()
    print("loss:",loss.item())
    for param in model.parameters():
        if param.shape[0] > 1e3:
            pass
        else:
            continue
        # print(param.shape)
        param_norm = param.grad.norm() if param.grad is not None else None
        print(param.abs().max(),param_norm)
    # for group in optimizer.param_groups:
    #     print(len(group['params']))
    #     for p in group['params']:
    #         if p.grad is None:
    #             continue
    #         print(p.grad.data)
    #         print(optimizer.state[p])
            # print(p.grad.data)
    # optimizer.step()




    # print(encoding.scaler)

    # xmin,xmax,ymin,ymax = 0,1,0,1
    # x1_list = np.linspace(xmin, xmax, 1000)
    # x2_list = np.linspace(ymin, ymax, 1000)
    # X1,X2 = np.meshgrid(x1_list,x2_list)
    # X1,X2 = X1.reshape(-1,1),X2.reshape(-1,1)
    # boundary_boolean = (X1 == xmin)|(X1 == xmax)|(X2 == ymin)|(X2 == ymax)
    # X_B = torch.tensor(np.concatenate((X1[boundary_boolean].reshape(-1,1),
    #     X2[boundary_boolean].reshape(-1,1)),axis = 1)).float().to(device)
    # res = encoding(X_B)[999,:,:]
    # print(res.shape,res)
    # print(encoding(x).shape)
    # for p in encoding.parameters():
    #     print(p[0].shape)
    '''
    