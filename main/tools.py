import torch
import numpy as np
import os
import json

# Calculate vectors
def dv_calculation(v1,v2):
    d1 = len(v1)
    d2 = len(v2)
    if d1 != d2:
        raise Exception("The vectors should have the same size.")
    dv = (v2-v1).abs()
    if (dv>0).sum() != 1:
        raise Exception("v2-v1 should be on the same direction as an axis.")
    return dv,d1

# Generate random sample points on the line v2-v1
def random_points_1D(n_points,v1,v2,device = 'cuda'):
    dv,d1 = dv_calculation(v1,v2)
    points = torch.rand((n_points,d1))
    points = (v1 + points*dv).to(device)
    return points

# Generate random sample points on the plane v2-v1,v3-v1
def random_points_2D(n_points,v1,v2,v3,device = 'cuda'):
    dv1,d1 = dv_calculation(v1,v2)
    dv2,d2 = dv_calculation(v1,v3)
    points = torch.rand((n_points,d1))
    points = (v1 + points*dv1 + points*dv2).to(device)
    return points

# Generate random sample points in the box v2-v1,v3-v1,v4-v1
def random_points_3D(n_points,v1,v2,v3,v4,device = 'cuda'):
    dv1,d1 = dv_calculation(v1,v2)
    dv2,d2 = dv_calculation(v1,v3)
    dv3,d3 = dv_calculation(v1,v4)
    points = torch.rand((n_points,d1))
    points = (v1 + points*dv1 + points*dv2 + points*dv3).to(device)
    return points

# Generate random sample points in the box v2-v1,v3-v1,v4-v1
def random_points_4D(n_points,v1,v2,v3,v4,v5,device = 'cuda'):
    dv1,d1 = dv_calculation(v1,v2)
    dv2,d2 = dv_calculation(v1,v3)
    dv3,d3 = dv_calculation(v1,v4)
    dv4,d4 = dv_calculation(v1,v5)
    points = torch.rand((n_points,d1))
    points = (v1 + points*dv1 + points*dv2 + points*dv3 + points*dv4).to(device)
    return points

# Generate random sample points on the line v2-v1
def collocation_points_1D(n_points,v1,v2,device = 'cuda'):
    dv,d1 = dv_calculation(v1,v2)
    points = torch.linspace(0,1,n_points).unsqueeze(-1)
    points = (v1 + points*dv).to(device)
    return points


# Generate folder
def my_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# RMSE
def my_rmse(X1,X2):
    return np.sqrt(((X1-X2)**2).mean())

# relative RMSE
def my_relativeL2(pred,true):
    error = pred-true
    return np.sqrt((error**2).sum()/(true**2).sum())

def plot_diff(x,ground_truth,field_name,f_name,f_path = "figs",eps = 1e-4,ifsave = False,ifplot = True,
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
    plt.title("{} - real\n(NRMSE: {})".format(field_name,nrmse))
    plt.colorbar(ticks=ct)
    plt.axis('off')
    fig.tight_layout()
    if ifsave:
        plt.savefig(f_path + '/{}.png'.format(f_name),dpi = 300)
        plt.close()
    elif ifplot:
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


def get_loss_curve(task_name,dir_name,main_path,output_config = False):
    task_path = os.path.join(main_path, "Tasks",task_name)

    result_path = os.path.join(task_path, dir_name)
    with open(result_path + ".json", "r") as config_file:
        config = json.load(config_file)
        # print(config)
    time = []
    error = []
    for result_file in os.listdir(result_path):
        if not result_file.endswith(".txt"):
            continue
        loss_curve = np.loadtxt(os.path.join(result_path, result_file))
        time.append(loss_curve[:,0])
        error.append(loss_curve[:,-1])

    time = np.array(time).mean(axis=0)
    error = np.array(error)
    x = time
    y = np.mean(error,axis=0)
    y1 = np.min(error,axis=0)
    y2 = np.max(error,axis=0)
    res_dict = {"x":x,"y":y,"y1":y1,"y2":y2}
    if output_config:
        return res_dict,config
    else:
        return res_dict

def get_lambda_curve(task_name,main_path):
    task_path = os.path.join(main_path, "Tasks",task_name)
    lambda_list = []
    error_list = []
    for dir_name in os.listdir(task_path):
        if dir_name.endswith(".json"):
            continue
        result_path = os.path.join(task_path, dir_name)
        with open(result_path + ".json", "r") as config_file:
            config = json.load(config_file)
            lambda_list.append(config["loss"]["lambda"])
        # time = []
        error = []
        for result_file in os.listdir(result_path):
            if not result_file.endswith(".txt"):
                continue
            loss_curve = np.loadtxt(os.path.join(result_path, result_file))
            error.append(loss_curve[-1,-1])
        error_list.append([np.mean(error),np.min(error),np.max(error)])

    lambda_list = np.array(lambda_list)
    error_list = np.array(error_list)[np.argsort(lambda_list)]
    lambda_list = lambda_list[np.argsort(lambda_list)]

    x = lambda_list
    y = error_list[:,0]
    y1 = error_list[:,1]
    y2 = error_list[:,2]
    res_dict = {"x": x,"y": y,"y1": y1,"y2": y2}
    return res_dict


def get_result_table(task_name,dict_path_list,main_path):
    task_path = os.path.join(main_path, "Tasks",task_name)
    result_dict = {}
    for dir_name in os.listdir(task_path):
        if dir_name.endswith(".json"):
            continue
        result_path = os.path.join(task_path, dir_name)
        result_name = result_path
        result_dict[result_name] = {}
        with open(result_path + ".json", "r") as config_file:
            config = json.load(config_file)
            for dict_path in dict_path_list:
                res = config
                for target in dict_path:
                    res = res[target]
                result_dict[result_name][target] = res
        time = []
        error = []
        for result_file in os.listdir(result_path):
            if not result_file.endswith(".txt"):
                continue
            loss_curve = np.loadtxt(os.path.join(result_path, result_file))
            error.append(loss_curve[-1,-1])
            time.append(loss_curve[-1,0])
        result_dict[result_name]["error"] = np.mean(error)
        result_dict[result_name]["time"] = np.mean(time)
        result_dict[result_name]["error_min"] = np.min(error)
        result_dict[result_name]["error_max"] = np.max(error)

    return result_dict