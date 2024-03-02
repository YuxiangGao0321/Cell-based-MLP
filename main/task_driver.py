import os
import json
from tools import my_mkdir
from PDE import Poisson_equation,multiscale_equation, Phase_field_equation_1d
from Solvers import Grid_MLP, PINN, PINN_energy
from tools import plot_diff

equation_name = "Phase_field_1d" #"Multiscale" # "Poisson_DirichletBC"
task_name = "GridMLP_decoupled" # 'GridMLP_decoupled_resolution_levels_network_sin' #"PINN_energy_lambda_ADAM" #'GridMLP_standard' # "PINN_lambda_ADAM" #"GridMLP_decoupled" #"PINN_lambda_ADAM_sin"
#'GridMLP_standard' #"PINN_energy_lambda_ADAM_eps=1" # "GridMLP_decoupled" # "PINN_energy_lambda_ADAM_eps=1" #"PINN_lambda_ADAM_eps=1"
task_path = "D:/Research_CAE/MyTinyCUDANN/tiny-cuda-nn/main/{}/Tasks/{}".format(equation_name,task_name)

if equation_name == "Poisson_DirichletBC":
    equation = Poisson_equation
    boundary_name_list = ["left","bottom","right","top"]
elif equation_name == "Multiscale":
    equation = multiscale_equation
    boundary_name_list = ["left","bottom","right","top"]
elif equation_name == "Phase_field_1d":
    equation = Phase_field_equation_1d
    boundary_name_list = ["left","right"]


task_length = len(os.listdir(task_path))
task_num = 0
for config_file_name in os.listdir(task_path):
    if not config_file_name.endswith(".json"):
        continue
    with open(os.path.join(task_path,config_file_name)) as f:
        config = json.load(f)
    result_path = os.path.join(task_path,config_file_name.split(".")[0])
    task_num += 1
    if os.path.exists(result_path) and len(os.listdir(result_path))>0:
        continue
    my_mkdir(result_path)
    n_test = config["experiment"]["n_test"]
    for i_test in range(n_test):
        print("[{}]/[{}]:".format(task_num,task_length),equation_name,task_name,config_file_name, i_test)
        if "PINN" in task_name:
            model_name = "MLP"
            if "energy" in task_name:
                solver = PINN_energy(config,equation())
            else:
                solver = PINN(config,equation())
            if "ADAM" in task_name:
                solver.train_adam(boundary_name_list = boundary_name_list)
            else:
                solver.train_lbfgs(boundary_name_list = boundary_name_list)
        elif "GridMLP" in task_name:
            model_name = "GridMLP"
            solver = Grid_MLP(config,equation())
            if "decoupled" in task_name:
                solver.train_decoupled(boundary_name_list = boundary_name_list)
            elif "standard" in task_name:
                solver.train_one_step(boundary_name_list = boundary_name_list)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        pred,u_real_plot = solver.eval_model_for_plot(501)

        plot_diff(pred,u_real_plot,field_name=model_name,f_name=i_test,f_path=result_path,
                  ifsave=True,ifplot=False)
        solver.save_test_loss(file_name=i_test,folder_path=result_path)
    