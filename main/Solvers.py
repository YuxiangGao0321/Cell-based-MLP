import sys
sys_path = 'D:/Research_CAE/MyTinyCUDANN/tiny-cuda-nn/main'
sys.path.append(sys_path)
from my_tiny_cuda import my_MLP,my_sin,weights_init_uniform
from my_tiny_cuda import plot_diff,my_relativeL2

from network import Squeeze
from gradient import grad1, grad2
from tools import random_points_1D,collocation_points_1D
import torch
import numpy as np
import time
import os

import tinycudann as tcnn


class Solver:
    def __init__(self,model,device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.MSE = torch.nn.MSELoss().to(self.device)
    def sample_all_boundary(self,batch_size_BC,field_min = 0, field_max = 1):
        n00 = torch.tensor([field_min,field_min])
        n01 = torch.tensor([field_min,field_max])
        n10 = torch.tensor([field_max,field_min])
        n11 = torch.tensor([field_max,field_max])
        X_bot = random_points_1D(int(batch_size_BC),n00,n10)
        X_left = random_points_1D(int(batch_size_BC),n00,n01)
        X_right = random_points_1D(int(batch_size_BC),n10,n11)
        X_top = random_points_1D(int(batch_size_BC),n01,n11)
        X_boundaries = torch.cat((X_bot,X_left,X_top,X_right), dim = 0)
        return X_boundaries
    
    def sample_boundaries(self,boundary_name_list,n_points_per_boundary,field_min = 0, field_max = 1):
        n00 = torch.tensor([field_min,field_min])
        n01 = torch.tensor([field_min,field_max])
        n10 = torch.tensor([field_max,field_min])
        n11 = torch.tensor([field_max,field_max])
        X_boundaries = []
        for name in boundary_name_list:
            if "bottom" in name:
                X = random_points_1D(n_points_per_boundary,n00,n10)
            elif "left" in name:
                X = random_points_1D(n_points_per_boundary,n00,n01)
            elif "right" in name:
                X = random_points_1D(n_points_per_boundary,n10,n11)
            elif "top" in name:
                X = random_points_1D(n_points_per_boundary,n01,n11)
            else:
                raise NotImplementedError
            X_boundaries.append(X)
        X_boundaries = torch.cat(X_boundaries, dim = 0)
        return X_boundaries
    
    def sample_periodic_boundary(self,batch_size_BC,field_min = 0, field_max = 1):
        n00 = torch.tensor([field_min,field_min])
        n01 = torch.tensor([field_min,field_max])
        n10 = torch.tensor([field_max,field_min])
        n11 = torch.tensor([field_max,field_max])
        X_bot = collocation_points_1D(int(batch_size_BC/4),n00,n10,device=self.device)
        X_left = collocation_points_1D(int(batch_size_BC/4),n00,n01,device=self.device)
        X_right = collocation_points_1D(int(batch_size_BC/4),n10,n11,device=self.device)
        X_top = collocation_points_1D(int(batch_size_BC/4),n01,n11,device=self.device)
        points_dict = {"bottom":X_bot,"left":X_left,"top":X_top,"right":X_right}
        return points_dict

    def generate_grid_points(self,resolution, field_min = 0, field_max = 1):
        x1_list = np.linspace(field_min, field_max, resolution)
        x2_list = np.linspace(field_min, field_max, resolution)
        X1,X2 = np.meshgrid(x1_list,x2_list)
        X_field = torch.tensor(np.concatenate((X1.reshape(-1,1),X2.reshape(-1,1)),
        axis = 1)).float().to(self.device)
        return X_field

    def eval_model(self, X_field, fixed_point = None):
        self.model.eval()
        with torch.no_grad():
            if fixed_point is not None:
                u_pred = (self.model(X_field)-self.model(fixed_point)).to('cpu').detach().numpy()
            else:
                u_pred = self.model(X_field).to('cpu').detach().numpy()
            u_error = my_relativeL2(u_pred,self.u_real)          
        return u_error


    def eval_model_for_plot(self,resolution, fixed_point = None):
        X_field = self.generate_grid_points(resolution)        
        u_real_plot = self.get_real_solution(X_field).reshape(resolution,resolution)
        self.model.eval()
        with torch.no_grad():
            if fixed_point is not None:
                pred = (self.model(X_field)-self.model(fixed_point)).to('cpu').detach().numpy().reshape(resolution,resolution)
            else:
                pred = self.model(X_field).to('cpu').detach().numpy().reshape(resolution,resolution)
        self.model.train()
        return pred,u_real_plot
    
    def get_real_solution(self,X_field):
        u_real = self.PDE.real_solution(X_field).to('cpu').detach().numpy()
        return u_real


    def save_test_loss(self,file_name,folder_path = None):
        path = os.path.join(folder_path,"{}.txt".format(file_name))
        np.savetxt(path, np.array(self.test_loss))

    



class PINN(Solver):
    def __init__(self,config,PDE,device = None):
        self.PDE = PDE
        self.config = config
        print(self.config)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        activation_name = config['network']['activation']
        if activation_name == 'Tanh':
            activation = torch.nn.Tanh()
        elif activation_name == "SiLU":
            activation = torch.nn.SiLU()
        elif activation_name == "Sin":
            activation = my_sin()
        else:
            print(activation_name)
            raise Exception("activation name error!")

        n_hidden = config['network']['n_hidden_layers']
        n_neurons = config['network']['n_neurons']
        model = my_MLP(activation = activation, n_input_dims = 2,
                    n_hidden = n_hidden, width = n_neurons,
                    spectral_norm = False,dtype = torch.float32).to(device)
        if activation_name == "Sin":
            model.apply(weights_init_uniform)
        super().__init__(model,device)
        self.model = model
    
        self.X_field = self.generate_grid_points(501)
        self.u_real = self.get_real_solution(self.X_field)
    
    def train_lbfgs(self,boundary_name_list = ["left","bottom","right","top"]):
        batch_size_BC = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(batch_size_BC/len(boundary_name_list))
        # X_boundaries = self.sample_all_boundary(batch_size_BC).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        batch_size = self.config['training']["interior_batch"]
        # X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

        lam = self.config["loss"]['lambda']

        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        max_iter = self.config["optimizer"]['max_iter']
        gamma=self.config["optimizer"]['gamma']

        n_steps = int(n_steps/max_iter)
        n_step_output = max(int(n_step_output/max_iter),1)
        n_step_decay = max(int(n_step_decay/max_iter),1)

        optimizer = torch.optim.LBFGS(self.model.parameters(),lr = self.config["optimizer"]['learning_rate'],
                            max_iter = max_iter,line_search_fn="strong_wolfe")

        
        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)
        diff_info = grad2(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)
            def closure():
                global bc_loss, inner_loss
                optimizer.zero_grad()
                # Boundary
                
                bc_loss = self.MSE(self.model(X_boundaries),f_boundaries)

                grad_result = diff_info.forward(X_I)

                PDE_residual = self.PDE.strong_form(X_I, grad_result)
                inner_loss = (PDE_residual**2).mean()
                loss = inner_loss + lam * bc_loss
                loss.backward()
                return loss
            
            optimizer.step(closure)

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()
            
            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma           

    def train_adam(self,boundary_name_list = ["left","bottom","right","top"]):
        batch_size_BC = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(batch_size_BC/len(boundary_name_list))
        # X_boundaries = self.sample_all_boundary(batch_size_BC).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        batch_size = self.config['training']["interior_batch"]
        X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

        lam = self.config["loss"]['lambda']

        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']


        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.config["optimizer"]['learning_rate'])
        
        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)
        diff_info = grad2(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        self.test_loss = []
        start = time.time()
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            
            optimizer.zero_grad()
            # Boundary
            
            bc_loss = self.MSE(self.model(X_boundaries),f_boundaries)

            grad_result = diff_info.forward(X_I)

            PDE_residual = self.PDE.strong_form(X_I, grad_result)
            inner_loss = (PDE_residual**2).mean()
            loss = inner_loss + lam * bc_loss
            loss.backward()
            
            optimizer.step()

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma

class PINN_energy(Solver):
    def __init__(self,config,PDE,device = None):
        self.PDE = PDE
        self.config = config
        print(self.config)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        activation_name = config['network']['activation']
        if activation_name == 'Tanh':
            activation = torch.nn.Tanh()
        elif activation_name == "SiLU":
            activation = torch.nn.SiLU()
        elif activation_name == "Sin":
            activation = my_sin()
        else:
            print(activation_name)
            raise Exception("activation name error!")

        n_hidden = config['network']['n_hidden_layers']
        n_neurons = config['network']['n_neurons']
        model = my_MLP(activation = activation, n_input_dims = 2,
                    n_hidden = n_hidden, width = n_neurons,
                    spectral_norm = False,dtype = torch.float32).to(device)
        if activation_name == "Sin":
            model.apply(weights_init_uniform)
        super().__init__(model,device)
        self.model = model
    
        self.X_field = self.generate_grid_points(501)
        self.u_real = self.get_real_solution(self.X_field)
    
    def train_lbfgs(self,boundary_name_list = ["left","bottom","right","top"]):
        batch_size_BC = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(batch_size_BC/len(boundary_name_list))
        # X_boundaries = self.sample_all_boundary(batch_size_BC).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        batch_size = self.config['training']["interior_batch"]
        # X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

        lam = self.config["loss"]['lambda']

        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        max_iter = self.config["optimizer"]['max_iter']
        gamma=self.config["optimizer"]['gamma']

        n_steps = int(n_steps/max_iter)
        n_step_output = max(int(n_step_output/max_iter),1)
        n_step_decay = max(int(n_step_decay/max_iter),1)

        optimizer = torch.optim.LBFGS(self.model.parameters(),lr = self.config["optimizer"]['learning_rate'],
                            max_iter = max_iter,line_search_fn="strong_wolfe")

        
        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)
        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)
            def closure():
                global bc_loss, inner_loss
                optimizer.zero_grad()
                # Boundary
                
                du_dx,du_dy,u = diff_info.forward_2d(X_I)


                inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
                bc_loss = self.MSE(self.model(X_boundaries),f_boundaries)
                loss = inner_loss + lam * bc_loss
                loss.backward()
                return loss
            
            optimizer.step(closure)

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()
            
            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma           

    def train_adam(self,boundary_name_list = ["left","bottom","right","top"]):
        batch_size_BC = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(batch_size_BC/len(boundary_name_list))
        # X_boundaries = self.sample_all_boundary(batch_size_BC).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        batch_size = self.config['training']["interior_batch"]
        X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

        lam = self.config["loss"]['lambda']

        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']


        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.config["optimizer"]['learning_rate'])
        
        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)
        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        self.test_loss = []
        start = time.time()
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            
            optimizer.zero_grad()
            # Boundary
            
            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            bc_loss = self.MSE(self.model(X_boundaries),f_boundaries)
            loss = inner_loss + lam * bc_loss
            loss.backward()
            
            optimizer.step()

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma

class PINN_energy_pbc(Solver):
    def __init__(self,config,PDE,device = None):
        self.PDE = PDE
        self.config = config
        print(self.config)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        activation_name = config['network']['activation']
        if activation_name == 'Tanh':
            activation = torch.nn.Tanh()
        elif activation_name == "SiLU":
            activation = torch.nn.SiLU()
        elif activation_name == "Sin":
            activation = my_sin()
        else:
            print(activation_name)
            raise Exception("activation name error!")

        n_hidden = config['network']['n_hidden_layers']
        n_neurons = config['network']['n_neurons']
        model = my_MLP(activation = activation, n_input_dims = 2,
                    n_hidden = n_hidden, width = n_neurons,
                    spectral_norm = False,dtype = torch.float32).to(device)
        if activation_name == "Sin":
            model.apply(weights_init_uniform)
        super().__init__(model,device)
        self.model = model
    
        self.X_field = self.generate_grid_points(501)
        self.u_real = self.get_real_solution(self.X_field)

        self.x00 = torch.zeros([1,2],dtype=torch.float32, device = self.device)
    
    def train_lbfgs(self):
        batch_size_BC = self.config['training']["boundary_batch"]
        self.pbc_collocation = self.sample_periodic_boundary(batch_size_BC)
        batch_size = self.config['training']["interior_batch"]
        # X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

        lam = self.config["loss"]['lambda']

        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        max_iter = self.config["optimizer"]['max_iter']
        gamma=self.config["optimizer"]['gamma']

        n_steps = int(n_steps/max_iter)
        n_step_output = max(int(n_step_output/max_iter),1)
        n_step_decay = max(int(n_step_decay/max_iter),1)

        optimizer = torch.optim.LBFGS(self.model.parameters(),lr = self.config["optimizer"]['learning_rate'],
                            max_iter = max_iter,line_search_fn="strong_wolfe")

        
        
        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)
            def closure():
                global bc_loss, inner_loss
                optimizer.zero_grad()
                # Boundary
                
                du_dx,du_dy,u = diff_info.forward_2d(X_I)


                inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
                bc_loss = self.pbc_loss()
                loss = inner_loss + lam * bc_loss
                loss.backward()
                return loss
            
            optimizer.step(closure)

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field,fixed_point=self.x00)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()
            
            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma           

    def train_adam(self):
        batch_size_BC = self.config['training']["boundary_batch"]
        self.pbc_collocation = self.sample_periodic_boundary(batch_size_BC)
        batch_size = self.config['training']["interior_batch"]
        X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

        lam = self.config["loss"]['lambda']

        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']


        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.config["optimizer"]['learning_rate'])
        
        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        self.test_loss = []
        start = time.time()
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            
            optimizer.zero_grad()
            # Boundary
            
            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            bc_loss = self.pbc_loss()
            loss = inner_loss + lam * bc_loss
            loss.backward()
            
            optimizer.step()

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field,fixed_point=self.x00)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma

    def pbc_loss(self):
        loss = self.MSE(self.model(self.pbc_collocation["left"]),self.model(self.pbc_collocation["right"])) + \
            self.MSE(self.model(self.pbc_collocation["bottom"]),self.model(self.pbc_collocation["top"]))
        return loss


class Grid_MLP(Solver):    
    def __init__(self,config,PDE,device = None):
        self.PDE = PDE
        self.config = config
        print(self.config)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        activation_name = config['network']['activation']
        if activation_name == 'Tanh':
            activation = torch.nn.Tanh()
        elif activation_name == "SiLU":
            activation = torch.nn.SiLU()
        elif activation_name == "Sin":
            activation = my_sin()
        else:
            print(activation_name)
            raise Exception("activation name error!")

        n_hidden = config['network']['n_hidden_layers']
        n_neurons = config['network']['n_neurons']
        spectral_norm = True if config['network']['spectral_norm'] == 1 else False
        self.mlp = my_MLP(activation = activation, n_input_dims = int(config["encoding"]["n_levels"]*config["encoding"]["n_features_per_level"]),
                    n_hidden = n_hidden, width = n_neurons,
                    spectral_norm = spectral_norm,dtype = torch.float32).to(device)
        if activation_name == "Sin":
            self.mlp.apply(weights_init_uniform)
        
        self.encoding = tcnn.Encoding1(2, config["encoding"],dtype=torch.float32)
        model = torch.nn.Sequential(self.encoding,self.mlp)
        super().__init__(model,device)
        self.model = model
    
        self.X_field = self.generate_grid_points(501)
        self.u_real = self.get_real_solution(self.X_field)

    def train_decoupled(self,boundary_name_list = ["left","bottom","right","top"]):
        pretrain_batch_size = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(pretrain_batch_size/len(boundary_name_list))
        n_step_output_pretrain = self.config['pretrain']['n_step_output']
        n_step_decay_pretrain = self.config["pretrain"]['n_step_decay']
        n_step_pretrain = self.config['pretrain']["n_steps"]+1

        # X_boundaries = self.sample_all_boundary(pretrain_batch_size).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        if_pretrain = False
        try:
            if_pretrain = self.config["pretrain"]["if_interior"]
        except:
            pass
        if if_pretrain == "True":
            print("Add interior points for pretrain")
            X_I = torch.rand([pretrain_batch_size, 2],dtype=torch.float32, device = self.device)
            X_boundaries = torch.cat((X_boundaries,X_I), dim = 0)
            n_step_pretrain = (n_step_pretrain-1)*2 + 1
            n_step_decay_pretrain = (n_step_decay_pretrain)*2

        optimizer_pretrain = torch.optim.Adam([
            {'params':self.encoding.parameters()},
            {'params':self.mlp.parameters(),'weight_decay':1e-6},
        ], lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)


        self.model.train()
        total_time = 0
        start = time.time()
        # Train for BC
        for i in range(1, n_step_pretrain):
            
            optimizer_pretrain.zero_grad()

            loss = self.MSE(self.model(X_boundaries),f_boundaries)
            
            loss.backward()
            optimizer_pretrain.step()
            
            if i%n_step_output_pretrain == 0:
                end = time.time()
                total_time += end - start
                print('Iter:',i,'loss_pretrain:',loss.item())
                start = time.time()
            
            if i%n_step_decay_pretrain == 0:
                for _ in optimizer_pretrain.param_groups:
                    _['lr'] = _['lr']/2

        # Get boundary values
        grid_values = {}
        with torch.no_grad():
            for name, p in self.encoding.named_parameters():
                for fixed_bounary_name in boundary_name_list:
                    if fixed_bounary_name in name:
                        grid_values[fixed_bounary_name] = p
                # if name == 'params_left':
                #     # grid_left = p
                #     grid_values["left"] = p
                # elif name == 'params_bottom':
                #     # grid_bottom = p
                #     grid_values["bottom"] = p
                # elif name == 'params_right':
                #     # grid_right = p
                #     grid_values["right"] = p
                # elif name == 'params_top':
                #     # grid_top = p
                #     grid_values["top"] = p

        # New grids
        self.encoding = tcnn.Encoding1(2, self.config["encoding"],dtype=torch.float32)
        self.model = torch.nn.Sequential(self.encoding,self.mlp)
        opti_group = []
        with torch.no_grad():
            for name, p in self.encoding.named_parameters():
                for fixed_bounary_name in boundary_name_list:
                    if fixed_bounary_name in name:
                        p[:] = grid_values[fixed_bounary_name][:]
                        p.requires_grad = False
                if p.requires_grad:
                    opti_group.append({'params':p})
                # elif name == 'params_left':
                #     p[:] = grid_left[:]
                #     p.requires_grad = False
                # elif name == 'params_bottom':
                #     p[:] = grid_bottom[:]
                #     p.requires_grad = False
                # elif name == 'params_right':
                #     p[:] = grid_right[:]
                #     p.requires_grad = False 
                # elif name == 'params_top':
                #     p[:] = grid_top[:]
                #     p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        self.mlp.eval()

        # Train for PDE
        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']

        batch_size = self.config['training']["interior_batch"]

        optimizer = torch.optim.Adam(opti_group, lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        batch_size = int(np.sqrt(batch_size))**2
        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.encoding.train()
        start = time.time()
        self.test_loss = []
        X_I_base = self.generate_grid_points(int(np.sqrt(batch_size)), field_min = 0, field_max = 0.95)
        X_I = X_I_base + torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)*0.05
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):

            # X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)
            

            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            loss = inner_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                

            if i % n_step_output == 0:
                X_I = X_I_base + torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)*0.05
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',loss.item(),"\n",'u_L2:',u_error,)
                self.encoding.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma

    def train_one_step(self,boundary_name_list = ["left","bottom","right","top"]):
        # One step training (Train for BC and PDE together)
        batch_size_BC = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(batch_size_BC/len(boundary_name_list))
        # X_boundaries = self.sample_all_boundary(batch_size_BC).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        batch_size = self.config['training']["interior_batch"]
        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']
        lam = self.config["loss"]['lambda']

        optimizer = torch.optim.Adam([
            {'params':self.encoding.parameters()},
            {'params':self.mlp.parameters(),'weight_decay':1e-6},
        ], lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)
        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            optimizer.zero_grad()
            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            bc_loss = self.MSE(self.model(X_boundaries),f_boundaries)
            loss = inner_loss + lam * bc_loss

            
            loss.backward()
            optimizer.step()
                

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma

    def train_pbc(self):
        self.x00 = torch.zeros([1,2],dtype=torch.float32, device = self.device)
        # One step training (Train for BC and PDE together)
        batch_size_BC = self.config['training']["boundary_batch"]
        self.pbc_collocation = self.sample_periodic_boundary(batch_size_BC)
        
        batch_size = self.config['training']["interior_batch"]
        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']
        lam = self.config["loss"]['lambda']

        optimizer = torch.optim.Adam([
            {'params':self.encoding.parameters()},
            {'params':self.mlp.parameters(),'weight_decay':1e-6},
        ], lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):
            optimizer.zero_grad()
            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            bc_loss = self.pbc_loss()
            loss = inner_loss + lam * bc_loss

            
            loss.backward()
            optimizer.step()
                

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field, fixed_point=self.x00)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',inner_loss.item(),"\n",
              'bc_loss:',bc_loss.item(),'u_L2:',u_error,)
                self.model.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma
    
    def pbc_loss(self):
        loss = self.MSE(self.model(self.pbc_collocation["left"]),self.model(self.pbc_collocation["right"])) + \
            self.MSE(self.model(self.pbc_collocation["bottom"]),self.model(self.pbc_collocation["top"]))
        return loss
    
    def train_decoupled_single_level(self,boundary_name_list = ["left","bottom","right","top"]):
        self.model = torch.nn.Sequential(self.encoding,Squeeze())
        pretrain_batch_size = self.config['training']["boundary_batch"]
        n_points_per_boundary = int(pretrain_batch_size/len(boundary_name_list))
        n_step_output_pretrain = self.config['pretrain']['n_step_output']
        n_step_decay_pretrain = self.config["pretrain"]['n_step_decay']
        n_step_pretrain = self.config['pretrain']["n_steps"]+1

        # X_boundaries = self.sample_all_boundary(pretrain_batch_size).to(self.device)
        X_boundaries = self.sample_boundaries(boundary_name_list,n_points_per_boundary).to(self.device)
        if_pretrain = False
        try:
            if_pretrain = self.config["pretrain"]["if_pretrain"]
        except:
            pass
        if if_pretrain == "True":
            print("Add interior points for pretrain")
            X_I = torch.rand([pretrain_batch_size, 2],dtype=torch.float32, device = self.device)
            X_boundaries = torch.cat((X_boundaries,X_I), dim = 0)
            n_step_pretrain = (n_step_pretrain-1)*2 + 1
            n_step_decay_pretrain = (n_step_decay_pretrain)*2

        optimizer_pretrain = torch.optim.Adam([
            {'params':self.encoding.parameters()},
        ], lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        f_boundaries = self.PDE.BC_function(X_boundaries).to(self.device)


        self.model.train()
        total_time = 0
        start = time.time()
        # Train for BC
        for i in range(1, n_step_pretrain):
            
            optimizer_pretrain.zero_grad()

            loss = self.MSE(self.model(X_boundaries),f_boundaries)
            
            loss.backward()
            optimizer_pretrain.step()
            
            if i%n_step_output_pretrain == 0:
                end = time.time()
                total_time += end - start
                print('Iter:',i,'loss_pretrain:',loss.item())
                start = time.time()
            
            if i%n_step_decay_pretrain == 0:
                for _ in optimizer_pretrain.param_groups:
                    _['lr'] = _['lr']/2

        # Get boundary values
        grid_values = {}
        with torch.no_grad():
            for name, p in self.encoding.named_parameters():
                for fixed_bounary_name in boundary_name_list:
                    if fixed_bounary_name in name:
                        grid_values[fixed_bounary_name] = p
                # if name == 'params_left':
                #     # grid_left = p
                #     grid_values["left"] = p
                # elif name == 'params_bottom':
                #     # grid_bottom = p
                #     grid_values["bottom"] = p
                # elif name == 'params_right':
                #     # grid_right = p
                #     grid_values["right"] = p
                # elif name == 'params_top':
                #     # grid_top = p
                #     grid_values["top"] = p

        # New grids
        self.encoding = tcnn.Encoding1(2, self.config["encoding"],dtype=torch.float32)
        self.model = self.model = torch.nn.Sequential(self.encoding,Squeeze())
        opti_group = []
        with torch.no_grad():
            for name, p in self.encoding.named_parameters():
                for fixed_bounary_name in boundary_name_list:
                    if fixed_bounary_name in name:
                        p[:] = grid_values[fixed_bounary_name][:]
                        p.requires_grad = False
                if p.requires_grad:
                    opti_group.append({'params':p})
                # elif name == 'params_left':
                #     p[:] = grid_left[:]
                #     p.requires_grad = False
                # elif name == 'params_bottom':
                #     p[:] = grid_bottom[:]
                #     p.requires_grad = False
                # elif name == 'params_right':
                #     p[:] = grid_right[:]
                #     p.requires_grad = False 
                # elif name == 'params_top':
                #     p[:] = grid_top[:]
                #     p.requires_grad = False

        # Train for PDE
        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']

        batch_size = self.config['training']["interior_batch"]

        optimizer = torch.optim.Adam(opti_group, lr=self.config["optimizer"]['learning_rate'], eps=1e-15)


        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.encoding.train()
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):

            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            loss = inner_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',loss.item(),"\n",'u_L2:',u_error,)
                self.encoding.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma



class Grid_MLP_pbc(Solver):    
    def __init__(self,config,PDE,device = None):
        self.PDE = PDE
        self.config = config
        print(self.config)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        activation_name = config['network']['activation']
        if activation_name == 'Tanh':
            activation = torch.nn.Tanh()
        elif activation_name == "SiLU":
            activation = torch.nn.SiLU()
        elif activation_name == "Sin":
            activation = my_sin()
        else:
            print(activation_name)
            raise Exception("activation name error!")

        n_hidden = config['network']['n_hidden_layers']
        n_neurons = config['network']['n_neurons']
        spectral_norm = True if config['network']['spectral_norm'] == 1 else False
        self.mlp = my_MLP(activation = activation, n_input_dims = int(config["encoding"]["n_levels"]*config["encoding"]["n_features_per_level"]),
                    n_hidden = n_hidden, width = n_neurons,
                    spectral_norm = spectral_norm,dtype = torch.float32).to(device)
        if activation_name == "Sin":
            self.mlp.apply(weights_init_uniform)
        
        self.encoding = tcnn.Encoding_allpbc(2, config["encoding"],dtype=torch.float32)
        model = torch.nn.Sequential(self.encoding,self.mlp)
        super().__init__(model,device)
        self.model = model
    
        self.X_field = self.generate_grid_points(501)
        self.u_real = self.get_real_solution(self.X_field)

        self.x00 = torch.zeros([1,2],dtype=torch.float32, device = self.device)

    def train_share_parameters(self):

        optimizer = torch.optim.Adam([
            {'params':self.encoding.parameters()},
            {'params':self.mlp.parameters(),'weight_decay':1e-6},
        ], lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        # Train for PDE
        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']

        batch_size = self.config['training']["interior_batch"]


        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):

            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            loss = inner_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field, fixed_point=self.x00)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',loss.item(),"\n",'u_L2:',u_error,)
                self.encoding.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma

    def train_share_parameters_single_level(self):
        self.model = torch.nn.Sequential(self.encoding,Squeeze())
        optimizer = torch.optim.Adam([
            {'params':self.encoding.parameters()},
        ], lr=self.config["optimizer"]['learning_rate'], eps=1e-15)

        # Train for PDE
        n_steps = self.config['training']['n_steps']
        n_step_output = self.config['training']['n_step_output']
        n_step_decay = self.config["optimizer"]['n_step_decay']
        gamma=self.config["optimizer"]['gamma']

        batch_size = self.config['training']["interior_batch"]


        diff_info = grad1(self.model, batch_size)
        diff_info.to_device(self.device)

        self.model.train()
        total_time = 0
        start = time.time()
        self.test_loss = []
        for i in range(0, n_steps+1): #for i in range(1, n_steps+1):

            X_I = torch.rand([batch_size, 2],dtype=torch.float32, device = self.device)

            du_dx,du_dy,u = diff_info.forward_2d(X_I)


            inner_loss = self.PDE.variational_energy(X_I,u,du_dx,du_dy).mean()
            loss = inner_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                

            if i % n_step_output == 0:
                end = time.time()
                total_time += end - start
                u_error = self.eval_model(self.X_field, fixed_point=self.x00)
                self.test_loss.append([total_time, i, u_error])
                print('Iter:',i,'inner_loss:',loss.item(),"\n",'u_L2:',u_error,)
                self.encoding.train()
                start = time.time()

            if i%n_step_decay == 0:
                for _ in optimizer.param_groups:
                    _['lr'] = _['lr'] * gamma


if __name__ == '__main__':
    from PDE import Poisson_equation as equation
    import json
    import os
    sys_path = os.path.dirname(os.path.abspath(__file__))
    # task_path = os.path.join(sys_path,"Tasks/PINN_lambda/config_4.json")
    task_path = "D:/Research_CAE/MyTinyCUDANN/tiny-cuda-nn/main/Poisson_DirichletBC/Tasks/PINN_lambda_ADAM"
    with open(task_path + "/config_4.json") as f:
        config = json.load(f)
    
    solver = PINN(config,equation())
    # solver.train_lbfgs()
    solver.train_adam()
    pred,u_real_plot = solver.eval_model_for_plot(501)
    # print(pred.shape, u_real_plot.shape, np.mean(np.abs((pred-u_real_plot))))
    plot_diff(pred,u_real_plot,field_name="u",f_name=" ")

    
    