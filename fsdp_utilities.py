"""
    To keep basic_training a bit cleaner, 
    we put the utilities here; model loading,
    setting up FSDP, etc.
"""
import torch
import numpy as np
from functools import partial
from argparse import Namespace, ArgumentParser
import torch.distributed as td
from torch.utils.data import DistributedSampler
import os
import h5py
from typing import Union
from pathlib import Path
import torch.nn as nn
from uNet import BasicBlock, Lpt2NbodyNet
from data_utils import *
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp.wrap import (
   always_wrap_policy as wrap_policy,
   transformer_auto_wrap_policy,
   wrap
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
if hasattr(torch.distributed.algorithms._checkpoint.checkpoint_wrapper, "apply_activation_checkpointing"):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
    )
    fsdp_ckpt = True
else:
    print("WARNING: torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing not found")
    print("FSDP will not apply activation checkpointing")
    fsdp_ckpt = False

from torch.distributed.fsdp.fully_sharded_data_parallel import (FullyShardedDataParallel as FSDP,
                                                                CPUOffload,
                                                                MixedPrecision,
                                                                ShardingStrategy,
                                                                FullStateDictConfig,
                                                                BackwardPrefetch,
                                                                StateDictType
)

from functools import partial
rank = int(os.environ.get('PMI_RANK', '0'))
zero = rank == 0

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('--environment', type=str, default='local')
    parser.add_argument("cuda_visible_devices", type=str, default='0')
    # parser.add_argument("--cuda_cache_path", type=str,default="/home/scratch/siyuh/nv_ComputeCache")  # to fix issue with auton lab gpu
    parser.add_argument("-c", "--config_file_path", type=str, default='')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--sharding', type=str, default='no-shard', choices=['no-shard', 'hybrid-shard', 'full-shard', 'grad-shard','hybrid-full-shard'])
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--cpu_offload', action='store_true')
    parser.add_argument('--precision', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('--train_data', type=str, default='data/') #train.h5
    parser.add_argument('--val_data', type=str, default='data/')
    parser.add_argument('--aug', type=int, default=0)
    parser.add_argument('--model', type=str, default='UNET', choices=['UNET'])
    parser.add_argument('--base_data_path', type=str, default='/lcrc/project/cosmo_ai/dongx/Multi-GPU/capacity-test/512-data/')
    parser.add_argument('--meta_init', action='store_true')
    parser.add_argument('--activation_checkpointing', action='store_true')
    # parser.add_argument('--prompt', type=str, default=None)
    # parser.add_argument('--num_tokens', type=int, default=100)
    
    args = parser.parse_args()
    
    if zero:
        print('Running with parameters:')
        for k, v in vars(args).items():
            print(f'{k}: {v}')
    return args

def setup_environment(args: Namespace) -> Namespace:
    """
        Torch distributed init using env:// requires the correct environment variables to be set.
        Map from default environment variables to the ones used on Polaris. Or wherever.
        Easily extensible by mapping, eg, slurm environment variables to the ones used here.

    """
    if not torch.cuda.is_available():
        raise NotImplementedError("No CUDA? FSDP needs CUDA or accelerators")
    if args.environment == 'pbs':
        os.environ['RANK'] = os.environ['PMI_RANK']# global 
        os.environ['LOCAL_RANK'] = os.environ['PMI_LOCAL_RANK'] # local
        os.environ['WORLD_SIZE'] = os.environ['PMI_SIZE']
        args.world_size = int(os.environ['PMI_SIZE'])
        args.global_rank = int(os.environ['PMI_RANK'])
        args.rank = args.global_rank
        args.local_rank = int(os.environ['PMI_LOCAL_RANK']) # wraparound since LOCAL_RANK is actually global?? WRT/ 
        args.local_size = int(os.environ['PMI_LOCAL_SIZE'])
        args.backend = 'nccl'
        args.num_nodes = args.world_size // args.local_size
    elif args.environment == 'local':
        # torch.backends.cuda.flash_sdp_enabled()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        args.rank = 0
        args.world_size = 1
        args.global_rank = 0
        args.local_rank = 0
        args.local_size = 1
        args.backend = 'nccl' 
    elif args.environment == 'slurm':
        os.environ['RANK'] = os.environ['SLURM_PROCID']# global
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID'] # local
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.global_rank = int(os.environ['SLURM_PROCID'])
        args.rank = args.global_rank
        args.local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
        args.local_rank = args.global_rank % args.local_size # wraparound since LOCAL_RANK is actually global?? WRT/
        args.backend = 'nccl'
        args.num_nodes = args.world_size // args.local_size
        # raise NotImplementedError("Slurm not implemented")
    return args


def init_distributed(args: Namespace) -> Namespace:
    print(f"{args.rank} init_distributed...")
    torch.cuda.set_device(args.local_rank)
    global_pg = td.init_process_group(backend=args.backend, init_method="env://")
    tot = torch.tensor(1).to(torch.cuda.current_device())
    # Use all-reduce to verify success here
    td.all_reduce(tot, op=td.ReduceOp.SUM)

    print(f"ARANK {args.rank}: Global rank {td.get_rank()} info: WORLD={td.get_world_size()}, has devices {torch.cuda.device_count()}, on device = {torch.cuda.current_device()}")
    print(f"ARANK {args.rank}: Local rank {args.local_rank}: {torch.cuda.current_device()}")
    print(f"ARANK {args.rank}: Total number of processes: {tot}")
    return args



def setup_model(args: Namespace) -> torch.nn.Module:
    """
        Initializing a model in FSDP is a bit esoteric and has a lot more 
        intricacy than a general model initialization. 
        Adding another model is straightforward and follows the same pattern as used here for GPT.
        The only intricacy is to import and correctly identify the transformer blocks for FSDP.
    """
    # Setup up configurations.  These are very similar to HF interface
    if args.model == 'UNET':
        # config = UNETConfig(
        #     block_size = args.seq_length, # configured in tokenizer to match GPT-3
        #     vocab_size = args.vocab_size,
        #     n_layer = args.num_layers,
        #     n_head = args.num_heads,
        #     n_embd = args.embed_size,
        #     dropout = args.dropout,
        # )
        arch = Lpt2NbodyNet
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    
    sharding = ShardingStrategy.NO_SHARD
    if args.sharding == 'full-shard':
        sharding = ShardingStrategy.FULL_SHARD
    elif args.sharding == 'grad-shard':
        sharding = ShardingStrategy.SHARD_GRAD_OP
    elif args.sharding == 'hybrid-full-shard':
        sharding = ShardingStrategy.HYBRID_SHARD
    elif args.sharding == 'hybrid-grad-shard':
        sharding = ShardingStrategy._HYBRID_SHARD_ZERO2

    # set up mixed precision
    prec = torch.float32 if args.precision == 'float32' else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=prec,
        reduce_dtype=prec,
        buffer_dtype=prec
    )
    # set up auto wrapping policy
    # twrap_policy = partial(
    #         transformer_auto_wrap_policy,
    #         transformer_layer_cls={Block,}
    #     )

    # auto wrapping policy
    my_auto_wrap_policy = partial(
				size_based_auto_wrap_policy, min_num_params=100_000
				)

    # set up checkpointing, if supported.
    if fsdp_ckpt:
            ckpt_fn = lambda submod: isinstance(submod, BasicBlock)
            non_reent_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=args.cpu_offload,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT
            )



    # large models need to be materialized as they  are
    # created to prevent OOM errors
    if args.meta_init:
        with torch.device('meta'):
            model_arch = arch(BasicBlock)
    else:
        model_arch = arch(BasicBlock)

    model = FSDP(model_arch,
            auto_wrap_policy=my_auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding, #FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
            cpu_offload=CPUOffload(offload_params=args.cpu_offload),
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE, # bit faster async comms, bit higher memory
            limit_all_gathers=False,
            use_orig_params=True,
            forward_prefetch=True,

            )
    if args.activation_checkpointing and fsdp_ckpt:
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reent_wrapper, check_fn=ckpt_fn
        )

    return model



def get_dataloaders(args:Namespace):
    train_loader = SimuData(args, split='train')
    val_loader = SimuData(args, split='val')
    train_sampler = DistributedSampler(train_loader, shuffle=True, num_replicas = args.world_size, rank = args.rank)
    val_sampler = DistributedSampler(val_loader, shuffle=False, num_replicas = args.world_size, rank = args.rank)
    train_loader = DataLoader(train_loader, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_loader, batch_size=args.batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def save_restart_checkpoint(save_directory: Union[str, Path],
                            STR:str,
                            model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            
                            args: Namespace,
                            epochnum: int, 
                            ): #scheduler: torch.optim.lr_scheduler._LRScheduler,
    """
        special checkpoint where we'll save all relevant items for a restart, including optimizer states.  I hope.
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True if args.sharding != 'no-shard' else False, rank0_only=True)
    with FSDP.state_dict_type(
                model, 
                StateDictType.FULL_STATE_DICT, 
                save_policy
            ):
                cpu_state = model.state_dict()
    # call on every rank, only actually exists on rank 0
    if args.sharding != 'no-shard':
        opt_state = FSDP.optim_state_dict(model, optimizer) # specify group if using HYBRID_SHARD...
    else:
        opt_state = optimizer.state_dict()
    if args.rank == 0:
        if not os.path.exists(f"{save_directory}"):
            os.makedirs(f"{save_directory}")
        

        save_name = f"{save_directory}/{args.run_name}_epoch{epochnum:02d}.pt"
        saveargs = vars(args)
        
        if args.rank == 0:
            save_obj = dict(
                        model_state=cpu_state,
                        optimizer_state=opt_state,
                        
                        epoch_number=epochnum,
                        args=saveargs
            ) #scheduler_state=scheduler.state_dict(),
            torch.save(save_obj, save_name)


