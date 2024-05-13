import os
import json
from tools import my_mkdir
from PDE import microscale_Poisson_PBC
from Solvers import Grid_MLP_pbc,PINN_energy_pbc,Grid_MLP
from tools import plot_diff,save_field_result

equation_name = "microscale_pbc"
task_name = "GridMLP_share_single_level"#'GridMLP_share_resolution_levels_sin_network_3e4' #"PINN_energy_lambda_ADAM_sin" #
#'GridMLP_standard' #"PINN_energy_lambda_ADAM_eps=1" # "GridMLP_decoupled" # "PINN_energy_lambda_ADAM_eps=1" #"PINN_lambda_ADAM_eps=1"
task_path = "D:/Research_CAE/MyTinyCUDANN/tiny-cuda-nn/main/{}/Tasks/{}".format(equation_name,task_name)

if equation_name == "microscale_pbc":
    equation = microscale_Poisson_PBC

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
                solver = PINN_energy_pbc(config,equation())
            # else:
            #     solver = PINN(config,equation())
            if "ADAM" in task_name:
                solver.train_adam()
            else:
                solver.train_lbfgs()
        elif "GridMLP" in task_name:
            model_name = "GridMLP"
            
            if "share" in task_name:
                solver = Grid_MLP_pbc(config,equation())
                if "single_level" in task_name:
                    solver.train_share_parameters_single_level()
                else:
                    solver.train_share_parameters()
            elif "standard" in task_name:
                solver = Grid_MLP(config,equation())
                solver.train_pbc()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        pred,u_real_plot = solver.eval_model_for_plot(501,fixed_point=solver.x00)

        plot_diff(pred,u_real_plot,field_name=model_name,f_name=i_test,f_path=result_path,
                  ifsave=True,ifplot=False)
        save_field_result(pred,file_name = "pred_{}".format(i_test),folder_path = result_path,if_overwrite = False)
        save_field_result(u_real_plot,file_name = "real_solution_plot".format(i_test),folder_path = result_path,if_overwrite = False)
        solver.save_test_loss(file_name=i_test,folder_path=result_path)
    