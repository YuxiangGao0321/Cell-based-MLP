# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch.autograd.function import once_differentiable
from tinycudann_bindings import _C
import numpy as np

def _torch_precision(tcnn_precision):
	if tcnn_precision == _C.Precision.Fp16:
		return torch.half
	elif tcnn_precision == _C.Precision.Fp32:
		return torch.float
	else:
		raise ValueError(f"Unknown precision {tcnn_precision}")

def free_temporary_memory():
	_C.free_temporary_memory()

class _module_function(torch.autograd.Function):
	@staticmethod
	def forward(ctx, native_tcnn_module, input, params, loss_scale):
		# If no output gradient is provided, no need to
		# automatically materialize it as torch.zeros.
		ctx.set_materialize_grads(False)

		native_ctx, output = native_tcnn_module.fwd(input, params)
		ctx.save_for_backward(input, params, output)
		ctx.native_tcnn_module = native_tcnn_module
		ctx.native_ctx = native_ctx
		ctx.loss_scale = loss_scale

		return output

	@staticmethod
	def backward(ctx, doutput):
		if doutput is None:
			return None, None, None, None

		if not doutput.is_cuda:
			print("TCNN WARNING: doutput must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			doutput = doutput.cuda()

		input, params, output = ctx.saved_tensors
		input_grad, weight_grad = _module_function_backward.apply(ctx, doutput, input, params, output)

		return None, input_grad, weight_grad, None

class _module_function_backward(torch.autograd.Function):
	@staticmethod
	def forward(ctx, ctx_fwd, doutput, input, params, output):
		ctx.ctx_fwd = ctx_fwd
		ctx.save_for_backward(input, params, doutput)
		with torch.no_grad():
			scaled_grad = doutput * ctx_fwd.loss_scale
			input_grad, weight_grad = ctx_fwd.native_tcnn_module.bwd(ctx_fwd.native_ctx, input, params, output, scaled_grad)
			input_grad = None if input_grad is None else (input_grad / ctx_fwd.loss_scale)
			weight_grad = None if weight_grad is None else (weight_grad / ctx_fwd.loss_scale)
		return input_grad, weight_grad

	@staticmethod
	def backward(ctx, dinput_grad, dweight_grad):
		# NOTE: currently support:
		#       ✓   d(dL_dinput)_d(dL_doutput)  doutput_grad
		#       ✓   d(dL_dinput)_d(params)      weight_grad
		#       ✓   d(dL_dinput)_d(input)       input_grad
		#       x   d(dL_dparam)_d(...)
		input, params, doutput = ctx.saved_tensors
		# assert dweight_grad is None, "currently do not support 2nd-order gradients from gradient of grid"
		with torch.enable_grad():
			# NOTE: preserves requires_grad info (this function is in no_grad() context by default when invoking loss.backward())
			doutput = doutput * ctx.ctx_fwd.loss_scale
		with torch.no_grad():
			doutput_grad, weight_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
				ctx.ctx_fwd.native_ctx,
				input,
				params,
				dinput_grad,
				doutput
			)
			# NOTE: be cautious when multiplying and dividing loss_scale
			#       doutput_grad uses dinput_grad
			#       weight_grad  uses dinput_grad * doutput
			#       input_grad   uses dinput_grad * doutput
			weight_grad = None if weight_grad is None else (weight_grad / ctx.ctx_fwd.loss_scale)
			input_grad = None if input_grad is None else (input_grad / ctx.ctx_fwd.loss_scale)

		# ctx_fwd,   doutput,      input,      params,      output
		return None, doutput_grad, input_grad, weight_grad, None

class Module(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module, self).__init__()

		self.native_tcnn_module = self._native_tcnn_module()
		self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

		self.seed = seed
		initial_params = self.native_tcnn_module.initial_params(seed)

		self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		self.register_parameter(name="params", param=self.params)

		self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0

	def forward(self, x):
		if not x.is_cuda:
			print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			x = x.cuda()

		batch_size = x.shape[0]
		batch_size_granularity = int(_C.batch_size_granularity())
		padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

		x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])

		output = _module_function.apply(
			self.native_tcnn_module,
			x_padded.to(torch.float).contiguous(),
			self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			self.loss_scale
		)
		return output[:batch_size, :self.n_output_dims]

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
		self.native_tcnn_module = self._native_tcnn_module()

	def extra_repr(self):
		return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"



class Module1(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module1, self).__init__()

		self.native_tcnn_module = self._native_tcnn_module()
		self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

		self.seed = seed
		initial_params = self.native_tcnn_module.initial_params(seed)
		### Yuxiang 02/15/2023 ###
		# self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		# self.register_parameter(name="params", param=self.params)
		# Boundary points indces
		if self.n_input_dims == 2:
			# print(self.encoding_config)
			n_levels = self.encoding_config["n_levels"]
			p_l_s = self.encoding_config["per_level_scale"]
			base_res = self.encoding_config["base_resolution"]
			n_features = self.encoding_config["n_features_per_level"]
			# self.bc_side = self.encoding_config["bc_side"]

			scale = np.floor(np.power(p_l_s, np.arange(n_levels,dtype="d"), dtype="d")*base_res-1, dtype="d")
			max_pos_list = scale.astype(int)
			res_list = scale.astype(int)+1
			level_table_size = (np.ceil(res_list**2/8)*8*n_features).astype(int)
			level_start_idx = np.add.accumulate(np.concatenate([np.array([0]),level_table_size[:-1]]))

			left_idx = []
			right_idx = []
			bottom_idx = []
			top_idx = []
			for max_pos,res,start_idx in zip(max_pos_list,res_list,level_start_idx):
				# for each level
				all_pos = np.arange(max_pos+1)
				left_idx_level = all_pos * res
				right_idx_level = max_pos + all_pos * res
				bottom_idx_level = all_pos
				top_idx_level = all_pos + (max_pos) * res

				for level_idx,grid_idx in zip([left_idx_level,right_idx_level,bottom_idx_level,top_idx_level],
					[left_idx,right_idx,bottom_idx,top_idx]):
					idx = level_idx * n_features
					idx = start_idx + np.concatenate([idx + j for j in range(n_features)])
					grid_idx.append(idx)

			self.boundary_idx_left = np.concatenate(left_idx)
			self.boundary_idx_right = np.concatenate(right_idx)
			self.boundary_idx_bottom = np.concatenate(bottom_idx)
			self.boundary_idx_top = np.concatenate(top_idx)
			self.inner_idx = np.delete(np.arange(initial_params.shape[0]),np.concatenate(
				[self.boundary_idx_left,self.boundary_idx_right,self.boundary_idx_bottom,self.boundary_idx_top]))

		self.params = initial_params
		# self.register_parameter(name="params", param=self.params)
		self.params_top = torch.nn.Parameter(initial_params[self.boundary_idx_top], requires_grad=True)
		self.params_bottom = torch.nn.Parameter(initial_params[self.boundary_idx_bottom], requires_grad=True)
		self.params_left = torch.nn.Parameter(initial_params[self.boundary_idx_left], requires_grad=True)
		self.params_right = torch.nn.Parameter(initial_params[self.boundary_idx_right], requires_grad=True)
		self.params_inner = torch.nn.Parameter(initial_params[self.inner_idx], requires_grad=True)

		self.register_parameter(name="params_top", param=self.params_top)
		self.register_parameter(name="params_bottom", param=self.params_bottom)
		self.register_parameter(name="params_left", param=self.params_left)
		self.register_parameter(name="params_right", param=self.params_right)
		self.register_parameter(name="params_inner", param=self.params_inner)
			### Yuxiang 02/15/2023 ###

		self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0

	def forward(self, x):
		if not x.is_cuda:
			print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			x = x.cuda()

		batch_size = x.shape[0]
		batch_size_granularity = int(_C.batch_size_granularity())
		padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

		x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])

		### Yuxiang 02/15/2023 ###
		my_params = self.params.clone()
		my_params[self.inner_idx] = self.params_inner
		my_params[self.boundary_idx_top] = self.params_top
		my_params[self.boundary_idx_right] = self.params_right
		# if self.bc_side == 'left':
		my_params[self.boundary_idx_bottom] = self.params_bottom
		my_params[self.boundary_idx_left] = self.params_left
		# else:
		# 	my_params[self.boundary_idx_left] = self.params_left
		# 	my_params[self.boundary_idx_bottom] = self.params_bottom			
		### Yuxiang 02/15/2023 ###
		output = _module_function.apply(
			self.native_tcnn_module,
			x_padded.to(torch.float).contiguous(),
			### Yuxiang 02/15/2023 ###
			my_params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			### Yuxiang 02/15/2023 ###
			# self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			self.loss_scale
		)
		return output[:batch_size, :self.n_output_dims]

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
		self.native_tcnn_module = self._native_tcnn_module()

	def extra_repr(self):
		return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"


class Module_pbc(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module_pbc, self).__init__()

		self.native_tcnn_module = self._native_tcnn_module()
		self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

		self.seed = seed
		initial_params = self.native_tcnn_module.initial_params(seed)
		### Yuxiang 02/15/2023 ###
		# self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		# self.register_parameter(name="params", param=self.params)
		# Boundary points indces
		if self.n_input_dims == 2:
			# print(self.encoding_config)
			n_levels = self.encoding_config["n_levels"]
			p_l_s = self.encoding_config["per_level_scale"]
			base_res = self.encoding_config["base_resolution"]
			n_features = self.encoding_config["n_features_per_level"]
			self.bc_side = self.encoding_config["bc_side"]

			scale = np.floor(np.power(p_l_s, np.arange(n_levels,dtype="d"), dtype="d")*base_res-1, dtype="d")
			max_pos_list = scale.astype(int)
			res_list = scale.astype(int)+1
			level_table_size = (np.ceil(res_list**2/8)*8*n_features).astype(int)
			level_start_idx = np.add.accumulate(np.concatenate([np.array([0]),level_table_size[:-1]]))

			left_idx = []
			right_idx = []
			bottom_idx = []
			top_idx = []
			for max_pos,res,start_idx in zip(max_pos_list,res_list,level_start_idx):
				# for each level
				all_pos = np.arange(max_pos+1)
				left_idx_level = all_pos * res
				right_idx_level = max_pos + all_pos * res
				bottom_idx_level = all_pos
				top_idx_level = all_pos + (max_pos) * res

				for level_idx,grid_idx in zip([left_idx_level,right_idx_level,bottom_idx_level,top_idx_level],
					[left_idx,right_idx,bottom_idx,top_idx]):
					idx = level_idx * n_features
					idx = start_idx + np.concatenate([idx + j for j in range(n_features)])
					grid_idx.append(idx)

			self.boundary_idx_left = np.concatenate(left_idx)
			self.boundary_idx_right = np.concatenate(right_idx)
			self.boundary_idx_bottom = np.concatenate(bottom_idx)
			self.boundary_idx_top = np.concatenate(top_idx)
			self.inner_idx = np.delete(np.arange(initial_params.shape[0]),np.concatenate(
				[self.boundary_idx_left,self.boundary_idx_right,self.boundary_idx_bottom,self.boundary_idx_top]))

		self.params = initial_params
		# self.register_parameter(name="params", param=self.params)
		self.params_top = torch.nn.Parameter(initial_params[self.boundary_idx_top], requires_grad=True)
		self.params_bottom = torch.nn.Parameter(initial_params[self.boundary_idx_bottom], requires_grad=True)
		self.params_left = torch.nn.Parameter(initial_params[self.boundary_idx_left], requires_grad=True)
		self.params_right = torch.nn.Parameter(initial_params[self.boundary_idx_right], requires_grad=True)
		self.params_inner = torch.nn.Parameter(initial_params[self.inner_idx], requires_grad=True)

		self.register_parameter(name="params_top", param=self.params_top)
		self.register_parameter(name="params_bottom", param=self.params_bottom)
		self.register_parameter(name="params_left", param=self.params_left)
		self.register_parameter(name="params_right", param=self.params_right)
		self.register_parameter(name="params_inner", param=self.params_inner)
			### Yuxiang 02/15/2023 ###

		self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0

	def forward(self, x):
		if not x.is_cuda:
			print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			x = x.cuda()

		batch_size = x.shape[0]
		batch_size_granularity = int(_C.batch_size_granularity())
		padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

		x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])

		### Yuxiang 02/15/2023 ###
		my_params = self.params.clone()
		my_params[self.inner_idx] = self.params_inner
		my_params[self.boundary_idx_top] = self.params_bottom
		my_params[self.boundary_idx_bottom] = self.params_bottom
		my_params[self.boundary_idx_right] = self.params_right
		my_params[self.boundary_idx_left] = self.params_left
		# if self.bc_side == 'left':
		# 	my_params[self.boundary_idx_bottom] = self.params_bottom
		# 	my_params[self.boundary_idx_left] = self.params_left
		# else:
		# 	my_params[self.boundary_idx_left] = self.params_left
		# 	my_params[self.boundary_idx_bottom] = self.params_bottom			
		### Yuxiang 02/15/2023 ###
		output = _module_function.apply(
			self.native_tcnn_module,
			x_padded.to(torch.float).contiguous(),
			### Yuxiang 02/15/2023 ###
			my_params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			### Yuxiang 02/15/2023 ###
			# self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			self.loss_scale
		)
		return output[:batch_size, :self.n_output_dims]

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
		self.native_tcnn_module = self._native_tcnn_module()

	def extra_repr(self):
		return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"

class Module_allpbc(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module_allpbc, self).__init__()

		self.native_tcnn_module = self._native_tcnn_module()
		self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

		self.seed = seed
		initial_params = self.native_tcnn_module.initial_params(seed)
		### Yuxiang 02/15/2023 ###
		# self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		# self.register_parameter(name="params", param=self.params)
		# Boundary points indces
		if self.n_input_dims == 2:
			# print(self.encoding_config)
			n_levels = self.encoding_config["n_levels"]
			p_l_s = self.encoding_config["per_level_scale"]
			base_res = self.encoding_config["base_resolution"]
			n_features = self.encoding_config["n_features_per_level"]
			# self.bc_side = self.encoding_config["bc_side"]

			scale = np.floor(np.power(p_l_s, np.arange(n_levels,dtype="d"), dtype="d")*base_res-1, dtype="d")
			max_pos_list = scale.astype(int)
			res_list = scale.astype(int)+1
			level_table_size = (np.ceil(res_list**2/8)*8*n_features).astype(int)
			level_start_idx = np.add.accumulate(np.concatenate([np.array([0]),level_table_size[:-1]]))

			left_idx = []
			right_idx = []
			bottom_idx = []
			top_idx = []
			left_bottom_idx = []
			left_top_idx = []
			right_bottom_idx = []
			right_top_idx = []
			for max_pos,res,start_idx in zip(max_pos_list,res_list,level_start_idx):
				# for each level
				all_pos = np.arange(max_pos+1)
				left_idx_level = (all_pos * res)[1:-1]
				right_idx_level = (max_pos + all_pos * res)[1:-1]
				bottom_idx_level = (all_pos)[1:-1]
				top_idx_level = (all_pos + (max_pos) * res)[1:-1]
				left_bottom_idx_level = (all_pos * res)[:1]
				left_top_idx_level = (all_pos * res)[-1:]
				right_bottom_idx_level = (max_pos + all_pos * res)[:1]
				right_top_idx_level = (max_pos + all_pos * res)[-1:]

				for level_idx,grid_idx in zip([left_idx_level,right_idx_level,bottom_idx_level,top_idx_level,
					left_bottom_idx_level,left_top_idx_level,right_bottom_idx_level,right_top_idx_level],
					[left_idx,right_idx,bottom_idx,top_idx,
					left_bottom_idx,left_top_idx,right_bottom_idx,right_top_idx]):
					idx = level_idx * n_features
					idx = start_idx + np.concatenate([idx + j for j in range(n_features)])
					grid_idx.append(idx)

			self.boundary_idx_left = np.concatenate(left_idx)
			self.boundary_idx_right = np.concatenate(right_idx)
			self.boundary_idx_bottom = np.concatenate(bottom_idx)
			self.boundary_idx_top = np.concatenate(top_idx)

			self.boundary_idx_left_bottom = np.concatenate(left_bottom_idx)
			self.boundary_idx_left_top = np.concatenate(left_top_idx)
			self.boundary_idx_right_bottom = np.concatenate(right_bottom_idx)
			self.boundary_idx_right_top = np.concatenate(right_top_idx)
			self.inner_idx = np.delete(np.arange(initial_params.shape[0]),np.concatenate(
				[self.boundary_idx_left,self.boundary_idx_right,self.boundary_idx_bottom,self.boundary_idx_top,
			self.boundary_idx_left_bottom,self.boundary_idx_left_top,
			self.boundary_idx_right_bottom,self.boundary_idx_right_top]))

		self.params = initial_params
		# self.register_parameter(name="params", param=self.params)
		# self.params_top = torch.nn.Parameter(initial_params[self.boundary_idx_top], requires_grad=True)
		self.params_bottom = torch.nn.Parameter(initial_params[self.boundary_idx_bottom], requires_grad=True)
		self.params_left = torch.nn.Parameter(initial_params[self.boundary_idx_left], requires_grad=True)
		# self.params_right = torch.nn.Parameter(initial_params[self.boundary_idx_right], requires_grad=True)

		self.params_left_bottom = torch.nn.Parameter(initial_params[self.boundary_idx_left_bottom], requires_grad=True)
		# self.params_left_top = torch.nn.Parameter(initial_params[self.boundary_idx_left_top], requires_grad=True)
		# self.params_right_bottom = torch.nn.Parameter(initial_params[self.boundary_idx_right_bottom], requires_grad=True)				
		# self.params_right_top = torch.nn.Parameter(initial_params[self.boundary_idx_right_top], requires_grad=True)

		self.params_inner = torch.nn.Parameter(initial_params[self.inner_idx], requires_grad=True)

		# self.register_parameter(name="params_top", param=self.params_top)
		self.register_parameter(name="params_bottom", param=self.params_bottom)
		self.register_parameter(name="params_left", param=self.params_left)
		# self.register_parameter(name="params_right", param=self.params_right)

		self.register_parameter(name="params_left_bottom", param=self.params_left_bottom)
		# self.register_parameter(name="params_left_top", param=self.params_left_top)
		# self.register_parameter(name="params_right_bottom", param=self.params_right_bottom)
		# self.register_parameter(name="params_right_top", param=self.params_right_top)

		self.register_parameter(name="params_inner", param=self.params_inner)
			### Yuxiang 02/15/2023 ###

		self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == _C.Precision.Fp16 else 1.0

	def forward(self, x):
		if not x.is_cuda:
			print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			x = x.cuda()

		batch_size = x.shape[0]
		batch_size_granularity = int(_C.batch_size_granularity())
		padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

		x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])

		### Yuxiang 02/15/2023 ###
		my_params = self.params.clone()
		my_params[self.inner_idx] = self.params_inner
		my_params[self.boundary_idx_top] = self.params_bottom
		my_params[self.boundary_idx_bottom] = self.params_bottom
		my_params[self.boundary_idx_right] = self.params_left
		my_params[self.boundary_idx_left] = self.params_left

		my_params[self.boundary_idx_left_bottom] = self.params_left_bottom
		my_params[self.boundary_idx_left_top] = self.params_left_bottom
		my_params[self.boundary_idx_right_bottom] = self.params_left_bottom
		my_params[self.boundary_idx_right_top] = self.params_left_bottom			
		### Yuxiang 02/15/2023 ###
		output = _module_function.apply(
			self.native_tcnn_module,
			x_padded.to(torch.float).contiguous(),
			### Yuxiang 02/15/2023 ###
			my_params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			### Yuxiang 02/15/2023 ###
			# self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			self.loss_scale
		)
		return output[:batch_size, :self.n_output_dims]

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
		self.native_tcnn_module = self._native_tcnn_module()

	def extra_repr(self):
		return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"


class NetworkWithInputEncoding(Module):
	"""
	Input encoding, followed by a neural network.

	This module is more efficient than invoking individual `Encoding`
	and `Network` modules in sequence.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	The output tensor can be either of type `torch.float` or `torch.half`,
	depending on which performs better on the system.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	n_output_dims : `int`
		Determines the shape of output tensors as `[:, n_output_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	network_config: `dict`
		Configures the neural network. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	"""
	def __init__(self, n_input_dims, n_output_dims, encoding_config, network_config, seed=1337):
		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.encoding_config = encoding_config
		self.network_config = network_config

		super(NetworkWithInputEncoding, self).__init__(seed=seed)

	def _native_tcnn_module(self):
		return _C.create_network_with_input_encoding(self.n_input_dims, self.n_output_dims, self.encoding_config, self.network_config)

class Network(Module):
	"""
	Neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a tensor of shape `[:, n_output_dims]`.

	The output tensor can be either of type `torch.float` or `torch.half`,
	depending on which performs better on the system.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	n_output_dims : `int`
		Determines the shape of output tensors as `[:, n_output_dims]`
	network_config: `dict`
		Configures the neural network. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	"""
	def __init__(self, n_input_dims, n_output_dims, network_config, seed=1337):
		self.n_input_dims = n_input_dims
		self.n_output_dims = n_output_dims
		self.network_config = network_config

		super(Network, self).__init__(seed=seed)

	def _native_tcnn_module(self):
		return _C.create_network(self.n_input_dims, self.n_output_dims, self.network_config)

class Encoding1(Module1):
	"""
	Input encoding to a neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
	`self.n_output_dims` depends on `n_input_dims` and the configuration
	`encoding_config`.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	dtype: `torch.dtype`
		Precision of the output tensor and internal parameters. A value
		of `None` corresponds to the optimally performing precision,
		which is `torch.half` on most systems. A value of `torch.float`
		may yield higher numerical accuracy, but is generally slower.
		A value of `torch.half` may not be supported on all systems.
	"""
	def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config
		if dtype is None:
			self.precision = _C.preferred_precision()
		else:
			if dtype == torch.float32:
				self.precision = _C.Precision.Fp32
			elif dtype == torch.float16:
				self.precision = _C.Precision.Fp16
			else:
				raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

		super(Encoding1, self).__init__(seed=seed)

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)

class Encoding_pbc(Module_pbc):
	"""
	Input encoding to a neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
	`self.n_output_dims` depends on `n_input_dims` and the configuration
	`encoding_config`.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	dtype: `torch.dtype`
		Precision of the output tensor and internal parameters. A value
		of `None` corresponds to the optimally performing precision,
		which is `torch.half` on most systems. A value of `torch.float`
		may yield higher numerical accuracy, but is generally slower.
		A value of `torch.half` may not be supported on all systems.
	"""
	def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config
		if dtype is None:
			self.precision = _C.preferred_precision()
		else:
			if dtype == torch.float32:
				self.precision = _C.Precision.Fp32
			elif dtype == torch.float16:
				self.precision = _C.Precision.Fp16
			else:
				raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

		super(Encoding_pbc, self).__init__(seed=seed)

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)

class Encoding_allpbc(Module_allpbc):
	"""
	Input encoding to a neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
	`self.n_output_dims` depends on `n_input_dims` and the configuration
	`encoding_config`.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	dtype: `torch.dtype`
		Precision of the output tensor and internal parameters. A value
		of `None` corresponds to the optimally performing precision,
		which is `torch.half` on most systems. A value of `torch.float`
		may yield higher numerical accuracy, but is generally slower.
		A value of `torch.half` may not be supported on all systems.
	"""
	def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config
		if dtype is None:
			self.precision = _C.preferred_precision()
		else:
			if dtype == torch.float32:
				self.precision = _C.Precision.Fp32
			elif dtype == torch.float16:
				self.precision = _C.Precision.Fp16
			else:
				raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

		super(Encoding_allpbc, self).__init__(seed=seed)

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)



class Encoding(Module):
	"""
	Input encoding to a neural network.

	Takes a `torch.float` input tensor of shape `[:, n_input_dims]` and maps
	it to a `dtype` tensor of shape `[:, self.n_output_dims]`, where
	`self.n_output_dims` depends on `n_input_dims` and the configuration
	`encoding_config`.

	Parameters
	----------
	n_input_dims : `int`
		Determines the shape of input tensors as `[:, n_input_dims]`
	encoding_config: `dict`
		Configures the encoding. Possible configurations are documented at
		https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md
	seed: `int`
		Seed for pseudorandom parameter initialization
	dtype: `torch.dtype`
		Precision of the output tensor and internal parameters. A value
		of `None` corresponds to the optimally performing precision,
		which is `torch.half` on most systems. A value of `torch.float`
		may yield higher numerical accuracy, but is generally slower.
		A value of `torch.half` may not be supported on all systems.
	"""
	def __init__(self, n_input_dims, encoding_config, seed=1337, dtype=None):
		self.n_input_dims = n_input_dims
		self.encoding_config = encoding_config
		if dtype is None:
			self.precision = _C.preferred_precision()
		else:
			if dtype == torch.float32:
				self.precision = _C.Precision.Fp32
			elif dtype == torch.float16:
				self.precision = _C.Precision.Fp16
			else:
				raise ValueError(f"Encoding only supports fp32 or fp16 precision, but got {dtype}")

		super(Encoding, self).__init__(seed=seed)

		self.n_output_dims = self.native_tcnn_module.n_output_dims()

	def _native_tcnn_module(self):
		return _C.create_encoding(self.n_input_dims, self.encoding_config, self.precision)

