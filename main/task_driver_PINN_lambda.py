import os
import json
from tools import my_mkdir
from PDE import Poisson_equation
from Solvers import PINN
from tools import plot_diff
task_path = "D:/Research_CAE/MyTinyCUDANN/tiny-cuda-nn/main/Poisson_DirichletBC/Tasks/PINN_lambda"

for config_file_name in os.listdir(task_path):
    if not config_file_name.endswith(".json"):
        continue
    with open(os.path.join(task_path,config_file_name)) as f:
        config = json.load(f)
    result_path = os.path.join(task_path,config_file_name.split(".")[0])
    my_mkdir(result_path)
    n_test = config["experiment"]["n_test"]
    for i_test in range(n_test):
        solver = PINN(config,Poisson_equation())
        solver.train_lbfgs()
        pred,u_real_plot = solver.eval_model_for_plot(501)

        plot_diff(pred,u_real_plot,field_name="MLP",f_name=i_test,f_path=result_path,
                  ifsave=True,ifplot=False)
        solver.save_test_loss(file_name=i_test,folder_path=result_path)
    