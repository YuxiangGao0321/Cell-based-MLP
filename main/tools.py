import torch
import numpy as np
import os
import json
import re
import torch.nn as nn
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
        os.makedirs(path)

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
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.size': 17})
    resolution = x.shape[0]
    xmin,ymin = 0,0
    xmax,ymax = L,L
    fig = plt.figure(figsize=(14,4))

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
    plt.title("Real solution")
    plt.colorbar(ticks=ct)
    # plt.axis('off')

    
    plt.subplot(133)
    diff = x - ground_truth
    # rmse = float(np.format_float_positional(my_rmse(x, ground_truth),precision = 4,unique=False, fractional=False, trim='k'))
    # nrmse = float(np.format_float_positional(my_relativeL2(x, ground_truth),precision = 4,
    #     unique=False, fractional=False, trim='k'))
    nrmse = my_relativeL2(x, ground_truth)
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

    # plt.title("{} - Real\n(NRMSE: {:.2e})".format(field_name,nrmse))
    plt.title("NRMSE: {:.2e}\n{} - Real".format(nrmse,field_name))
    color_bar = plt.colorbar(ticks=ct)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    color_bar.ax.yaxis.set_major_formatter(formatter)
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
    pattern = r'^(100|[1-9]?[0-9])\.txt$'
    task_path = os.path.join(main_path, "Tasks",task_name)

    result_path = os.path.join(task_path, dir_name)
    with open(result_path + ".json", "r") as config_file:
        config = json.load(config_file)
        # print(config)
    time = []
    error = []
    for result_file in os.listdir(result_path):
        # if not result_file.endswith(".txt"):
        #     continue
        # elif "pred" in result_file or "real" in result_file:
        #     continue
        if not re.match(pattern, result_file):
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
    pattern = r'^(100|[1-9]?[0-9])\.txt$'
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
            if not re.match(pattern, result_file):
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
            if not result_file.endswith(".txt") or "pred" in result_file or "real" in result_file:
                continue
            # print(result_file)
            loss_curve = np.loadtxt(os.path.join(result_path, result_file))
            error.append(loss_curve[-1,-1])
            time.append(loss_curve[-1,0])
        result_dict[result_name]["error"] = np.mean(error)
        result_dict[result_name]["time"] = np.mean(time)
        result_dict[result_name]["error_min"] = np.min(error)
        result_dict[result_name]["error_max"] = np.max(error)

    return result_dict

def get_lambda(json_path):
    with open(json_path, "r") as config_file:
        config = json.load(config_file)
    return config["loss"]["lambda"]

def save_field_result(field,file_name,folder_path = None,if_overwrite = False):
    path = os.path.join(folder_path,"{}.txt".format(file_name))
    if os.path.exists(path) and not if_overwrite:
        print("The file already exists.")
    else:
        np.savetxt(path, np.array(field))

def generate_grid_points(resolution, field_min = 0, field_max = 1, device = 'cuda'):
    x1_list = np.linspace(field_min, field_max, resolution)
    x2_list = np.linspace(field_min, field_max, resolution)
    X1,X2 = np.meshgrid(x1_list,x2_list)
    X_field = torch.tensor(np.concatenate((X1.reshape(-1,1),X2.reshape(-1,1)),
    axis = 1)).float().to(device)
    return X_field

def sample_all_boundary(batch_size_BC,field_min = 0, field_max = 1, device = 'cuda'):
    n00 = torch.tensor([field_min,field_min])
    n01 = torch.tensor([field_min,field_max])
    n10 = torch.tensor([field_max,field_min])
    n11 = torch.tensor([field_max,field_max])
    X_bot = random_points_1D(int(batch_size_BC),n00,n10, device = device)
    X_left = random_points_1D(int(batch_size_BC),n00,n01, device = device)
    X_right = random_points_1D(int(batch_size_BC),n10,n11, device = device)
    X_top = random_points_1D(int(batch_size_BC),n01,n11, device = device)
    X_boundaries = torch.cat((X_bot,X_left,X_top,X_right), dim = 0)
    return X_boundaries


class ShiftedLegendrePolynomialBatch(nn.Module):
    """
    A PyTorch module that represents a batch of shifted Legendre polynomials
    up to a specified maximum order on [0,1], along with their derivatives.

    For each order n, the shifted Legendre polynomial is defined as:
        P_n^shifted(x) = P_n(2x - 1),
    where P_n is the standard Legendre polynomial on [-1,1].

    Given an input tensor x, the module evaluates all P_n^shifted(x) (now
    including n=0) and their derivatives for n from 0 to max_order
    and returns them as tensors.
    """

    def __init__(self, max_order):
        """
        Args:
            max_order (int): The maximum order of Legendre polynomials to compute (>=1).
        """
        super().__init__()
        if not isinstance(max_order, int) or max_order < 1:
            raise ValueError("max_order must be an integer >= 1.")
        self.max_order = max_order

    def forward(self, x):
        """
        Evaluate shifted Legendre polynomials up to max_order at points x in [0,1],
        including the 0th polynomial.

        Args:
            x (torch.Tensor): Input tensor of shape (N,) or (N,1) with values in [0,1].

        Returns:
            torch.Tensor:
                Shape (N, max_order+1) containing [P_0^shifted(x), P_1^shifted(x), ..., P_max_order^shifted(x)].
        """
        # Ensure x is a 1D tensor
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() != 1:
            raise ValueError("Input tensor x must be of shape (N,) or (N,1).")

        # Shift x from [0,1] to z in [-1,1]
        z = 2.0 * x - 1.0  # Shape: (N,)

        # Compute polynomials P_0, P_1, ..., P_max_order
        P = self._compute_polynomials(z)  # Shape: (N, max_order+1)

        # Return all polynomials, including order 0
        return P

    def derivative(self, x):
        """
        Compute the derivatives of shifted Legendre polynomials up to max_order
        at points x in [0,1], including the 0th polynomial (which is always zero).

        d/dx [P_n^shifted(x)] = 2 * P_n'(z), where z = 2x - 1.

        Args:
            x (torch.Tensor): Input tensor of shape (N,) or (N,1) with values in [0,1].

        Returns:
            torch.Tensor:
                Shape (N, max_order+1) containing the derivatives
                [dP_0^shifted/dx, dP_1^shifted/dx, ..., dP_max_order^shifted/dx].
        """
        # Ensure x is a 1D tensor
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() != 1:
            raise ValueError("Input tensor x must be of shape (N,) or (N,1).")

        # Shift x from [0,1] to z in [-1,1]
        z = 2.0 * x - 1.0  # Shape: (N,)

        # Compute all polynomials P0 to P_max_order
        P = self._compute_polynomials(z)  # Shape: (N, max_order+1)

        # We will build the derivative array as (N, max_order+1).
        N = z.shape[0]
        dPdx = torch.zeros(N, self.max_order+1, device=z.device, dtype=z.dtype)

        # For n = 0, derivative is zero. We already initialized to zero.
        # For n >= 1, use the standard Legendre derivative formula:
        #   P_n'(z) = n/(1 - z^2) * [z P_n(z) - P_{n-1}(z)],
        #   then multiply by 2 for d/dx of the shifted polynomial.
        n = torch.arange(1, self.max_order+1, device=z.device, dtype=z.dtype)  # 1..max_order
        denominator = 1.0 - z**2 + 1e-9  # avoid divide-by-zero
        numer = z.unsqueeze(1) * P[:, 1:] - P[:, :-1]  # shape: (N, max_order)
        dPdz = (n / denominator.unsqueeze(1)) * numer  # shape: (N, max_order)

        # Chain rule: dP_n^shifted/dx = 2 * P_n'(z) for n>=1
        dPdx[:, 1:] = 2.0 * dPdz

        return - dPdx

    def _compute_polynomials(self, z):
        """
        Compute standard Legendre polynomials P0 to P_max_order at points z in [-1,1].

        Uses the recursive relation:
            P0(z) = 1
            P1(z) = z
            (k+1)*P_{k+1}(z) = (2k + 1)*z*P_k(z) - k*P_{k-1}(z)

        Args:
            z (torch.Tensor): Tensor of shape (N,) with values in [-1,1].

        Returns:
            torch.Tensor:
                Tensor of shape (N, max_order+1) containing
                [P0(z), P1(z), ..., P_max_order(z)].
        """
        N = z.shape[0]

        # Initialize with P0(z) = 1, P1(z) = z
        P_list = [torch.ones_like(z), z.clone()]  # P0, P1

        # Compute P2 to P_max_order using recursion
        for k in range(1, self.max_order):
            P_k = P_list[-1]    # P_k(z)
            P_km1 = P_list[-2]  # P_{k-1}(z)
            # (k+1) P_{k+1}(z) = (2k+1)*z*P_k(z) - k*P_{k-1}(z)
            P_kp1 = ((2.0*k + 1.0)*z*P_k - k*P_km1) / (k + 1.0)
            P_list.append(P_kp1)

        # Stack into shape: (N, max_order+1)
        P = torch.stack(P_list, dim=1)
        return P


class ShiftedLegendrePolynomial2D(nn.Module):
    """
    A PyTorch module for 2D shifted Legendre polynomials on [0,1] x [0,1].
    The 2D polynomial of orders (m, n) is given by:

        P_{m,n}^{(2D)}(x,y) = P_m^shifted(x) * P_n^shifted(y),

    where each P_k^shifted is the 1D shifted Legendre polynomial
    (mapped from [0,1] to [-1,1]).
    """

    def __init__(self, max_order):
        """
        Args:
            max_order (int): The maximum order for the polynomials in x and y.
        """
        super().__init__()
        self.max_order = max_order

        # Re-use the 1D Shifted Legendre class
        self.legendre_1d = ShiftedLegendrePolynomialBatch(max_order)

    def forward(self, x, y):
        """
        Compute 2D shifted Legendre polynomials up to (max_order, max_order),
        including the 0th polynomial in each dimension.

        Args:
            x (torch.Tensor): Shape (N,) or (N, 1), values in [0,1].
            y (torch.Tensor): Shape (N,) or (N, 1), values in [0,1].

        Returns:
            polynomials_2d (torch.Tensor):
                By default here, we return shape (N, (max_order+1)*(max_order+1)),
                which includes orders from (0,0) through (max_order, max_order).
        """
        # Get 1D polynomials for x and y (including the 0th term).
        Px = self.legendre_1d(x)  # shape: (N, max_order+1)
        Py = self.legendre_1d(y)  # shape: (N, max_order+1)

        # Outer product for each sample => shape: (N, max_order+1, max_order+1)
        Px_expanded = Px.unsqueeze(2)  # (N, max_order+1, 1)
        Py_expanded = Py.unsqueeze(1)  # (N, 1, max_order+1)
        polynomials_2d = Px_expanded * Py_expanded  # (N, max_order+1, max_order+1)

        # Optionally flatten to (N, (max_order+1)*(max_order+1))
        polynomials_2d = polynomials_2d.reshape(-1, (self.max_order+1)*(self.max_order+1))
        return polynomials_2d

    def derivatives(self, x, y):
        """
        Compute partial derivatives of the 2D polynomials wrt x and y:
            d/dx [P_m^shifted(x) * P_n^shifted(y)] = (d/dx P_m^shifted(x)) * P_n^shifted(y)
            d/dy [P_m^shifted(x) * P_n^shifted(y)] = P_m^shifted(x) * (d/dy P_n^shifted(y))

        Args:
            x, y (torch.Tensor): same shape constraints as forward().

        Returns:
            d2d_dx (torch.Tensor): shape (N, (max_order+1)*(max_order+1))
            d2d_dy (torch.Tensor): shape (N, (max_order+1)*(max_order+1))
        """
        # Evaluate 1D polynomials and derivatives (including 0th).
        Px = self.legendre_1d(x)         # (N, max_order+1)
        Py = self.legendre_1d(y)         # (N, max_order+1)
        dPx = self.legendre_1d.derivative(x)  # (N, max_order+1)
        dPy = self.legendre_1d.derivative(y)  # (N, max_order+1)

        # Compute partial wrt x:
        #   d/dx [Px(i,m)*Py(i,n)] = dPx(i,m) * Py(i,n)
        dPx_expanded = dPx.unsqueeze(2)  # (N, max_order+1, 1)
        Py_expanded  = Py.unsqueeze(1)   # (N, 1, max_order+1)
        d2d_dx = dPx_expanded * Py_expanded  # (N, max_order+1, max_order+1)

        # Compute partial wrt y:
        #   d/dy [Px(i,m)*Py(i,n)] = Px(i,m) * dPy(i,n)
        Px_expanded  = Px.unsqueeze(2)   # (N, max_order+1, 1)
        dPy_expanded = dPy.unsqueeze(1)  # (N, 1, max_order+1)
        d2d_dy = Px_expanded * dPy_expanded  # (N, max_order+1, max_order+1)

        # Flatten both
        d2d_dx = d2d_dx.reshape(-1, (self.max_order+1)*(self.max_order+1))
        d2d_dy = d2d_dy.reshape(-1, (self.max_order+1)*(self.max_order+1))

        return d2d_dx, d2d_dy



def test_derivatives_2d():
    """
    Compare the partial derivatives from the 'model.derivatives(...)'
    to those obtained via PyTorch autograd.
    """
    # 1. Create some example data
    torch.manual_seed(0)
    N = 5
    max_order = 3
    x = torch.rand(N, requires_grad=True)
    y = torch.rand(N, requires_grad=True)

    # 2. Instantiate the model
    model_2d = ShiftedLegendrePolynomial2D(max_order)

    # 3. Get the polynomials => shape (N, (max_order+1)^2)
    poly_2d = model_2d(x, y)

    # 4. We'll form a random linear combination of these polynomials to get a scalar function
    #    f_i = sum_{j}( random_coeffs[j] * poly_2d[i,j] )
    #    Then the partial derivative wrt x_i is sum_{j}( random_coeffs[j] * d/dx [poly_2d[i,j]] ).
    #    We'll compare that to autograd.
    coeffs = torch.randn(( (max_order+1)*(max_order+1), ), requires_grad=False)

    # f shape => (N,)
    f = (poly_2d * coeffs).sum(dim=1)  # sum_j [ poly_2d[i,j]*coeffs[j] ]

    # 5. Now do autograd: partial f wrt x => df_dx, partial f wrt y => df_dy
    #    Using `torch.autograd.grad`, we can get per-sample gradients.
    #    However, note that grad outputs are typically aggregated; we'll pass `grad_outputs=torch.ones_like(f)`
    #    so it does a vector-Jacobian product that yields shape (N,) for each partial.
    df_dx, df_dy = torch.autograd.grad(
        f, 
        [x, y],
        grad_outputs=torch.ones_like(f),
        create_graph=True
    )  # each is shape (N,)

    # 6. Compare with the model's own partial derivatives
    #    shape => d2d_dx, d2d_dy are (N, (max_order+1)^2)
    d2d_dx, d2d_dy = model_2d.derivatives(x, y)

    # The partial derivative wrt x of f_i is the dot-product
    #   sum_j[ d2d_dx[i, j] * coeffs[j] ]
    # Similarly for y.
    # => shape (N,)
    dx_dot = (d2d_dx * coeffs).sum(dim=1)
    dy_dot = (d2d_dy * coeffs).sum(dim=1)

    # 7. Print or compare the results
    print("AutoGrad partial wrt x:", df_dx)
    print("Analytic partial wrt x:", dx_dot)
    print("Difference (x):", (df_dx - dx_dot).abs().max().item())

    print("AutoGrad partial wrt y:", df_dy)
    print("Analytic partial wrt y:", dy_dot)
    print("Difference (y):", (df_dy - dy_dot).abs().max().item())

# Run the test
if __name__ == "__main__":
    test_derivatives_2d()
