import os
import json
from tools import my_mkdir
from PDE import microscale_Poisson_PBC
from Solvers import Grid_MLP_pbc,PINN_energy_pbc,Grid_MLP
from tools import plot_diff
import numpy as np

equation_name = "microscale_pbc"
task_name = "GridMLP_share"#'GridMLP_share_resolution_levels_sin_network_3e4' #"PINN_energy_lambda_ADAM_sin" #
#'GridMLP_standard' #"PINN_energy_lambda_ADAM_eps=1" # "GridMLP_decoupled" # "PINN_energy_lambda_ADAM_eps=1" #"PINN_lambda_ADAM_eps=1"
task_path = "D:/Research_CAE/MyTinyCUDANN/tiny-cuda-nn/main/{}/Tasks/{}".format(equation_name,task_name)

if equation_name == "microscale_pbc":
    equation = microscale_Poisson_PBC

task_length = 1
task_num = 0
config_file_name = "config_0_plot.json"


with open(os.path.join(task_path,config_file_name)) as f:
    config = json.load(f)
result_path = os.path.join(task_path,config_file_name.split(".")[0])
task_num += 1
# if os.path.exists(result_path) and len(os.listdir(result_path))>0:
#     continue
my_mkdir(result_path)
n_test = config["experiment"]["n_test"]
for i_test in range(n_test):
    print("[{}]/[{}]:".format(task_num,task_length),equation_name,task_name,config_file_name, i_test)
    pred_file = os.path.join(result_path,"pred_data_{}.txt".format(i_test))
    u_real_plot_file = os.path.join(result_path,"u_real_plot.txt")
    if os.path.exists(pred_file):
        pred = np.loadtxt(pred_file)
        u_real_plot = np.loadtxt(u_real_plot_file)
        model_name = "CellMLP"
    else:        
        if "PINN" in task_name:
            model_name = "MLP"
            if "energy" in task_name:
                solver = PINN_energy_pbc(config,equation())
            # else:
            #     solver = PINN(config,equation())
            if "ADAM" in task_name:
                solver.train_adam()
            else:
                solver.train_lbfgs()
        elif "GridMLP" in task_name:
            model_name = "CellMLP"
            
            if "share" in task_name:
                solver = Grid_MLP_pbc(config,equation())
                solver.train_share_parameters()
            elif "standard" in task_name:
                solver = Grid_MLP(config,equation())
                solver.train_pbc()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
        pred,u_real_plot = solver.eval_model_for_plot(501,fixed_point=solver.x00)
        np.savetxt(pred_file,pred)
        np.savetxt(u_real_plot_file,u_real_plot)

    plot_diff(pred,u_real_plot,field_name=model_name,f_name=i_test,f_path=result_path,
              ifsave=True,ifplot=False)
    # solver.save_test_loss(file_name=i_test,folder_path=result_path)
    