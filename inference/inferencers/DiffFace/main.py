from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments

if __name__ == "__main__":
    args = get_arguments()
    image_editor = ImageEditor(args)
    image_editor.edit_image_by_prompt()




# # torch pruning == 1.2.5 ??? If not -- downgrade it. At least it works as expected

# from typing import Callable, List, Union, Dict, TextIO, TypeVar, Optional, Union, Tuple
# from pathlib import Path
# import torch_pruning as tp
# import torch_integral as inn
# import torch
# from abc import ABC, abstractmethod
# import yaml
# import sys
# from pruning_jdd.pruning_utils import save_report
# from pruning_jdd.op_counter import count_ops_and_params
# from utils.logging import log_info, log_warning
# from copy import deepcopy
# from torch_integral import standard_continuous_dims, IntegralModel

# def to_device(x, device):
#     def _to_device(x, device):
#         if isinstance(x, (list, tuple)):
#             for v in x:
#                 _to_device(v, device)
#         elif isinstance(x, torch.Tensor):
#             x.data = x.data.to(device)
#     _to_device(x, device)
#     return x

# def _is_any_layer_mentioned(key, layers):
#     return any([
#         layer_name in key 
#         and
#         not f".{layer_name}" in key # <== to prevent same naming of modules and submodules
#         for layer_name in layers
#     ])



# PathType = Optional[Union[str, Path]]
# InOutType = Union[torch.Tensor, List[torch.Tensor]]
# ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)

# """
# Possible importance score functions
# """
# imp_fns = {
#     "taylor": tp.importance.TaylorImportance,
#     "magnitude": tp.importance.MagnitudeImportance,
#     'lamp': tp.importance.LAMPImportance,
#     "hessian": tp.importance.HessianImportance,
#     "bn_scale": tp.importance.BNScaleImportance,
#     "random": tp.importance.RandomImportance
# }



# class BasePruner(ABC):
#     def __init__(
#         self, 
#         experiment_path: PathType,
#         log_file: Optional[TextIO], 
#         loss_fn: Optional[Callable],
#         **kwargs
#     ):
#         self.experiment_path: PathType = Path(experiment_path) if experiment_path else None
#         self.log_file: Optional[TextIO] = log_file if log_file else sys.stdout
#         self.pruning_cfg: Dict = kwargs
#         self.pruner_cfg: Dict = self.pruning_cfg['pruner_cfg'].copy()
#         self.loss_fn = loss_fn
#         self.req_grad = False
#         self.stats = defaultdict()

#         if self.pruner_cfg.get('ch_sparsity_dict'):
#             self.pruner_cfg.pop('ch_sparsity')


#     @abstractmethod
#     def prune(self):
#         pass
    
#     @classmethod
#     def __get_sample_input(self, dataloader, device):
#         sample = next(iter(dataloader))

#         to_device(sample, device)
#         return sample

#     def __call__(self, model, dataloader, verbose=True, save_report=False, **kwargs):
#         # loss calculation before, parameters, flops, logging
#         model.train(False)
#         self.device = next(iter(model.parameters())).device
#         self.input_example, gt, *_ = self.__get_sample_input(dataloader, self.device)
        
#         with torch.no_grad():
#             self.stats['flops_before'], self.stats['total_params_before'] = count_ops_and_params(
#                 model=model, 
#                 example_inputs=self.input_example
#             )

#         self.stats['loss_before'] = Trainer._calculate_loss(
#             model=model, 
#             dataloader=dataloader,
#             req_grad=self.req_grad,
#             num_batches=self.pruning_cfg["num_batches"],
#             device=self.device
#         )

#         model = self.prune(model, dataloader=dataloader, **kwargs)

#         log_info("Calculating loss after pruning")
#         self.stats['loss_after'] = Trainer._calculate_loss(
#             model=model, 
#             dataloader=dataloader, 
#             req_grad=False, 
#             num_batches=self.pruning_cfg["num_batches"],
#             device=self.device
#         )

#         if isinstance(model, IntegralModel):
#             inn_model = model
#             model = inn_model.get_unparametrized_model()
#             model.integral = True
#             print(type(inn_model), type(model))
        

#         with torch.no_grad():
#             self.stats['flops_after'], self.stats['total_params_after'] = count_ops_and_params(
#                 model=model, 
#                 example_inputs=self.input_example
#             )
        
#         self.stats['flops_before'], self.stats['flops_after'] = (
#             self.stats['flops_before'] / 1e+9, 
#             self.stats['flops_after'] / 1e+9
#         )
#         self.stats['ratio'] = self.stats['total_params_before'] / self.stats['total_params_after']
#         if verbose:
#             log_info("\n".join([
#                 f"Loss before: {self.stats['loss_before']:.8f}",
#                 f"Loss after: {self.stats['loss_after']:.8f}",
#                 f"Total number of params before: {(self.stats['total_params_before'] / 1e+6):.4f}M",
#                 f"Total number of params after: {(self.stats['total_params_after'] / 1e+6):.4f}M",
#                 f"Flops: {self.stats['flops_before']:.5f} GFlops -> {self.stats['flops_after']:.5f} GFlops",
#                 f"Ratio: {self.stats['ratio']:.4f}x less params",
#             ]))
#         # loss calculation after, parameters, flops, logging        
#         if len(self.pruning_record) == 0:
#             log_warning("There is no pruning!")
#         elif save_report:
#             save_report(self.pruning_record, self.experiment_path, self.stats['ratio'])

#         if hasattr(model, 'integral'):
#             model = inn_model
        
#         model.train(True)
        
#         ###########################
#         ### Fast finetune after ###
#         ###########################
#         if self.pruning_cfg['finetune']:
#             assert kwargs.get('dataset', None), "Please, provide dataset for small finetuning after pruning"
#             trainer = Trainer(
#                 model=model, device=self.device, 
#                 trainer_config=self.pruning_cfg['trainer_config'],
#                 dataset=kwargs['dataset'], 
#                 custom_collate_fn=kwargs['custom_collate_fn']
#             )

#             trainer.train()
        
#         if self.pruning_cfg.get('use_discrete_model', True) and hasattr(model, 'integral'):
#             log_info("Converting the integral model into discrete one!")
#             model = model.get_unparametrized_model()
            
#         return model

#     @property
#     def get_last_stats(self):
#         return self.stats

#     def __repr__(self):
#         desc = "\n".join([
#             self.__class__.__name__,
#             yaml.dump(
#                 {k: v.__str__() for k, v in vars(self).items() if not isinstance(v, Callable)}
#             ),
#             "Pruning config:",
#             yaml.dump(self.pruning_cfg),
#             ]
#         )
#         return desc


# class Pruner(BasePruner):
#     def __init__(
#         self, 
#         experiment_path: PathType,
#         log_file: Optional[TextIO]=None,
#         loss_fn: Optional[Callable]=None,
#         **kwargs
#     ):
#         super(Pruner, self).__init__(experiment_path, log_file, loss_fn, **kwargs)

#         self.pruner_fn: Callable = tp.pruner.MetaPruner
#         self.imp_fn_name: str = self.pruner_cfg.pop("imp_fn", "magnitude")

#         assert self.imp_fn_name in imp_fns.keys(), f"{self.imp_fn_name} does not implemented!"
#         self.imp_fn: Callable = imp_fns[self.imp_fn_name]

#         if self.imp_fn in [
#             tp.importance.GroupTaylorImportance, tp.importance.TaylorImportance,
#             tp.importance.GroupHessianImportance, tp.importance.HessianImportance
#         ]:
#             log_info(f"{self.imp_fn.__name__} requires grad!")                    
#             self.req_grad = True

#         log_info(self)
        
        
#     def prune(
#         self, 
#         model: ModuleType,
#         dataloader: torch.utils.data.Dataset,
#         **kwargs
#     ) -> ModuleType:
#         if hasattr(self, "input_example"):
#             ...
#         else: # in case you use prune method, not __call__
#             self.device = next(iter(model.parameters())).device
#             self.input_example, *_ = self._BasePruner__get_sample_input(dataloader, self.device)
        

#         if self.req_grad:
#             for name, p in model.named_parameters():
#                 if not p.requires_grad:
#                     log_info(f"Be careful. Unfreezing {name}!") 
#                     p.requires_grad = True

#         if self.pruner_cfg.get('ch_sparsity_dict'):
#             self.pruner_cfg['ch_sparsity_dict'] = {
#                 model.get_submodule(k): v for k, v in self.pruner_cfg['ch_sparsity_dict'].items()
#             }        
        
#         if len(self.pruner_cfg.get("ignored_layers", [])):
#             log_info("Ignoring layers: ")
#             log_info(self.pruner_cfg["ignored_layers"])
#             self.pruner_cfg["ignored_layers"] = [model.get_submodule(k) for k in self.pruner_cfg["ignored_layers"]]


#         importance = self.imp_fn()
#         pruner = self.pruner_fn(
#             model,
#             example_inputs=self.input_example,
#             importance=importance,
#             **self.pruner_cfg
#         )           
        
#         #################################################
#         #################### Pruning ####################
#         #################################################
#         self.pruning_record = []
#         for group in pruner.step(interactive=True):
#             log_info(f'\n{group}')
#             for gr in group:
#                 dep, idxs = gr
#                 target_module = dep.target.module
#                 target_name = dep.target.name
#                 pruning_fn = dep.handler
#                 self.pruning_record.append(
#                     (target_name, target_module, target_module.__class__.__name__, pruning_fn, idxs)
#                 )

#             group.prune()
#         return model
    

# def _get_module_dim(model, key):
#     submodule = model.get_submodule(key.replace('.weight', '').replace('.bias', ''))

#     if isinstance(submodule, torch.nn.ConvTranspose2d):
#         return [1]

#     return [0]

# def get_processed_dims(model, input_layers, output_layers, pruning_dict, ignore_others=True):
#     all_continuous_dims = standard_continuous_dims(model)
#     continuous_dims = all_continuous_dims.copy()
#     discrete_dims = {}
#     for key in all_continuous_dims:
#         if _is_any_layer_mentioned(key, input_layers):
#             discrete_dims[key] = [1]
#             log_info(f"{key:_<30} is set to prune only outputs!")
#         elif _is_any_layer_mentioned(key, output_layers):
#             if 'bias' in key:
#                 discrete_dims[key] = continuous_dims.pop(key, [0])
#                 log_info(f"{key:_<30} will not be affected!")
#             else:
#                 discrete_dims[key] = [0]
#                 log_info(f"{key:_<30} is set to prune only inputs!")
            
#         if _is_any_layer_mentioned(key, pruning_dict):
#             log_info(f"{key:_<30} will be pruned")
#             continuous_dims[key] = _get_module_dim(model, key)
#         elif ignore_others:
#             log_info(f"{key:_<30} will be ignored")
#             continuous_dims.pop(key, [])
#         else:
#             ...

#     return continuous_dims, discrete_dims



# import functools
# def error_handler(func):
#     @functools.wraps(func)
#     def inner_function(*args,**kwargs):
#         try:
#             result = func(*args, **kwargs)
#         except Exception as err:
#             name = ' '.join(func.__name__.split('_'))
#             raise BaseException(f"Something went wrong in function --> {name}. Look here: {func.__module__}")
#         return result
#     return inner_function


# @error_handler
# def trace_model(model, continuous_dims, input_example, discrete_dims=None):
#     tracer = inn.IntegralTracer(model, continuous_dims=continuous_dims, discrete_dims=discrete_dims)
#     return tracer.build_groups(*input_example)




# from typing import Tuple
# ModuleType = TypeVar("ModuleType", bound=torch.nn.Module)
# InOutType = Union[torch.Tensor, Tuple[torch.Tensor]]


# class IntegralPruner(BasePruner):
#     def __init__(
#         self, 
#         experiment_path: PathType,
#         log_file: Optional[TextIO]=None,
#         loss_fn: Optional[Callable]=None,
#         **kwargs
#     ):
#         super(IntegralPruner, self).__init__(experiment_path, log_file, loss_fn, **kwargs)

#         self.input_layers = self.pruner_cfg['input_layers']
#         self.output_layers = self.pruner_cfg['output_layers']
#         self.leave_integral = self.pruner_cfg.get('leave_integral', [])
#         self.ignore_least = self.pruner_cfg.get('ignore_least', True)
#         self.permutation_config = self.pruner_cfg['permutation_config']
#         self.permutation_config['class'] = (
#             getattr(torch_integral.permutation, self.permutation_config['class']) 
#             if self.permutation_config.get('class', None) 
#             else torch_integral.permutation.NOptPermutation
#         )

#         self.pruning_dict = self.pruner_cfg['pruning_dict']

#         if isinstance(self.pruner_cfg['wrapper_cfg'], (list, tuple)):
#             self.pruner_cfg['wrapper_cfg'] = functools.reduce(
#                 lambda x, y: {**x, **y}, self.pruner_cfg['wrapper_cfg']
#             ) # <== in case the list of dicts is given (works well with .yaml's and current merging of arguments in universal framework)

#         log_info(self)
        

#     def prune(
#         self, 
#         model: ModuleType,
#         dataloader: torch.utils.data.DataLoader,
#         **kwargs
#     ) -> ModuleType:
#         if hasattr(self, "input_example"):
#             ...
#         else:
#             self.device = next(iter(model.parameters())).device
#             self.input_example, *_ = self._BasePruner__get_sample_input(dataloader, self.device)
        
#         continuous_dims, discrete_dims = get_processed_dims(
#             model, 
#             self.input_layers, 
#             self.output_layers,
#             self.pruning_dict, 
#             self.ignore_least
#         )


#         groups = trace_model(model.to('cpu'), continuous_dims, to_device(self.input_example, 'cpu'), discrete_dims) # <== get all groups

#         model = model.to(self.device)
#         wrapper = inn.IntegralWrapper( 
#             **self.pruner_cfg["wrapper_cfg"],
#             permutation_config=self.permutation_config,

#         )

#         self.pruning_record = []

#         inn_model = wrapper(model, self.input_example, continuous_dims, related_groups=groups)
#         display(Markdown("<h1>After conversion</h1>"))
#         check_model(inn_model, fp_original_model, num_samples=3)

#         for group in inn_model.groups:
#             # print(_is_any_layer_mentioned(group.parmas[0]))
#             group.new_size = next(
#                 (
#                     size for name, size in self.pruning_dict.items()
#                     if name in group.params[0]['name']
#                     # if any(name in param['name'] for param in group.params if param['dim'] == 0)
#                 ), 
#                 group.size # same size as before
#             )
#             self.pruning_record.append(
#                 (str(group).replace('\n', ''), group.size, group.new_size) # ! TODO: update pruning record similarly to the `Pruner`
#             )

#             log_info(f"\n{group}")
#             if group.size == group.new_size:
#                 log_info(f"\t[GROUP INFO] Group above will preserve size {group.new_size}, but has been converted to integral")
#             else:
#                 log_info(f"\t[GROUP INFO] Group above will be resampled to {group.new_size} and finetuned as integral")
#                 group.reset_distribution(inn.UniformDistribution(group.new_size, group.size))
#                 group.reset_grid(inn.TrainableGrid1D(group.new_size))
        
#         check_model(inn_model, fp_original_model, num_samples=3)
#         inn_model.grid_tuning(False, True, False)
#         return inn_modela