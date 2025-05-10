from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


# Author: Yongjun
class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return DifferentiableAllReduceSum.apply(grad_output), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, input_):
        world_size = get_tp_group().world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        # All-reduce-sum.
        torch.distributed.all_reduce(input_,
                                     group=get_tp_group().device_group,
                                     op=torch.distributed.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return DifferentiableIdentity.apply(grad_output), None


class DifferentiableAllGather(torch.autograd.Function):
    """All gather in a differentiable fashion"""

    @staticmethod
    def forward(ctx, input_):
        world_size = get_tp_group().world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty((world_size, ) + input_size,
                                    dtype=input_.dtype,
                                    device=input_.device,
                                    requires_grad=input_.requires_grad)
        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor,
                                                 input_,
                                                 group=get_tp_group().device_group)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return DifferentiableReduceScatterSum.apply(grad_output), None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    """Reduce scatter in a differentiable fashion"""

    @staticmethod
    def forward(ctx, input_):
        world_size = get_tp_group().world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_

        unsharded_batch_size, *rest_size = input_.shape
        assert unsharded_batch_size % world_size == 0
        # Allocate output tensor.
        output_tensor = torch.empty(unsharded_batch_size // world_size, *rest_size,
                                    device=input_.device,
                                    dtype=input_.dtype,
                                    requires_grad=input_.requires_grad)
        # Reduce-scatter-sum.
        torch.distributed.reduce_scatter_tensor(output_tensor,
                                                input_,
                                                group=get_tp_group().device_group,
                                                op=torch.distributed.ReduceOp.SUM)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        return DifferentiableAllGather.apply(grad_output), None


def differentiable_identity(input_: torch.Tensor):
    return DifferentiableIdentity.apply(input_)


def differentiable_all_reduce_sum(input_: torch.Tensor):
    return DifferentiableAllReduceSum.apply(input_)


def differentiable_all_gather(input_: torch.Tensor, dim: int = -1):
    world_size = get_tp_group().world_size
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    assert -input_.dim() <= dim < input_.dim(), (
        f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

    if dim < 0:
        # Convert negative dim to positive.
        dim += input_.dim()
    input_size = input_.size()

    output_tensor = DifferentiableAllGather.apply(input_)

    # Reshape
    output_tensor = output_tensor.movedim(0, dim)
    output_tensor = output_tensor.reshape(input_size[:dim] +
                                          (world_size *
                                          input_size[dim], ) +
                                          input_size[dim + 1:])
    return output_tensor
