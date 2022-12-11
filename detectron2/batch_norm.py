import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function
from torch.nn import functional as F

from detectron2.utils import comm

from .wrappers import BatchNorm2d


class BN2dFitlered(nn.Module):
            def __init__(self, Channels, Thres=4.0  ):
                super(BN2dFitlered , self).__init__()
                self.ChannelNum=Channels
                self.beta=nn.Parameter(torch.tensor([0.0]*int(Channels)).reshape(1,Channels,1,1), requires_grad=True)
                self.gamma=nn.Parameter(torch.tensor([1.0]*int(Channels)).reshape(1,Channels,1,1), requires_grad=True)
                self.Thres=Thres
                
            def forward(self, xorig):
                x=xorig.permute([1,0,2,3])
                x=x.reshape((self.ChannelNum,-1))
                
                Mean=torch.mean(x, dim=-1).reshape((self.ChannelNum,1))
                Var=torch.var(x, dim=-1).reshape((self.ChannelNum,1))
                Mean=Mean.expand((self.ChannelNum,x.shape[1]))
                Var=Var.expand((self.ChannelNum,x.shape[1]))
                
                eps=1e-10
                normalized= (x-Mean)/torch.sqrt(Var+eps)
                
                Selected= ((normalized<self.Thres) * (normalized>-self.Thres)).float()
                #masked mean
                Mean=torch.sum(x*Selected, dim=[-1])/torch.sum(Selected,dim=[-1])
                
                Diff=(x - Mean.reshape((self.ChannelNum,1)).expand(self.ChannelNum,x.shape[1])  )**2
                
                Mean=Mean.reshape((1,self.ChannelNum,1,1))
                Mean=Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                
                Var= torch.sum(Diff*Selected , dim=[-1])/torch.sum(Selected,dim=[-1])
                Var=Var.reshape((1,self.ChannelNum,1,1))
                Var=Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                eps=1e-10
                beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                bnfiltered= ((gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+beta
                return bnfiltered


class BN2dFitleredMoments(nn.Module):
            def __init__(self, Channels, Thres=4.0  ):
                super(BN2dFitlered , self).__init__()
                self.ChannelNum=Channels
                self.beta=nn.Parameter(torch.tensor([0.0]*int(Channels)).reshape(1,Channels,1,1), requires_grad=True)
                self.gamma=nn.Parameter(torch.tensor([1.0]*int(Channels)).reshape(1,Channels,1,1), requires_grad=True)
                self.Thres=Thres
                self.alpha=0.9
                self.Mean=torch.tensor([0.0]*int(Channels)) 
                self.Var=torch.tensor([1.0]*int(Channels))
                
            def forward(self, xorig):
                x=xorig.permute([1,0,2,3])
                x=x.reshape((self.ChannelNum,-1))
                
                Mean=torch.mean(x, dim=-1).reshape((self.ChannelNum,1))
                Var=torch.var(x, dim=-1).reshape((self.ChannelNum,1))
                Mean=Mean.expand((self.ChannelNum,x.shape[1]))
                Var=Var.expand((self.ChannelNum,x.shape[1]))
                
                eps=1e-10
                normalized= (x-Mean)/torch.sqrt(Var+eps)
                
                Selected= ((normalized<self.Thres) * (normalized>-self.Thres)).float()
                #masked mean
                Mean=torch.sum(x*Selected, dim=[-1])/torch.sum(Selected,dim=[-1])
                
                Diff=(x - Mean.reshape((self.ChannelNum,1)).expand(self.ChannelNum,x.shape[1])  )**2
                
                #Mean=Mean.reshape((1,self.ChannelNum,1,1))               
                Var= torch.sum(Diff*Selected , dim=[-1])/torch.sum(Selected,dim=[-1])
                #Var=Var.reshape((1,self.ChannelNum,1,1))
                
                self.Mean=self.alpha*self.Mean+(1.0-self.alpha)*Mean
                self.Var=self.alpha*self.Var+(1.0-self.alpha)*Var
        
                MeanExp=self.Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                VarExp=self.Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
        
                
                eps=1e-10
                beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                bnfiltered= ((gamma*(xorig-MeanExp))/torch.sqrt(VarExp+eps)      )+beta
                return bnfiltered

class BN1dFitlered(nn.Module):
    def __init__(self,Thres=4.0):
        super(BN1dFitlered , self).__init__()
        self.Thres=Thres
        self.beta=nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.gamma=nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def SetThreshold(Thres):
        self.Thres=Thres      
    def forward(self, xorig):
        x=xorig.view(-1)
        Mean=torch.mean(x)
        Var=torch.var(x)
        eps=1e-10
        normalized= (x-Mean)/torch.sqrt(Var+eps)
        Selected=(normalized<self.Thres) * (normalized>-self.Thres)
        prunedx=x[Selected]
        
        Mean=torch.mean(prunedx)
        Var=torch.var(prunedx)
        eps=1e-10
        bn= (self.gamma*(xorig-Mean)/torch.sqrt(Var+eps))+self.beta
        return bn

class BN1dFitleredMoments(nn.Module):
    def __init__(self,Thres=4.0):
        super(BN1dFitleredMoments , self).__init__()
        self.Thres=Thres
        self.beta=nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.gamma=nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.alpha=0.9
        self.Mean=0.0
        self.Var=1.0

    def SetThreshold(Thres):
        self.Thres=Thres      
    def forward(self, xorig):
        x=xorig.view(-1)
        Mean=torch.mean(x)
        Var=torch.var(x)
        eps=1e-10
        normalized= (x-Mean)/torch.sqrt(Var+eps)
        Selected=(normalized<self.Thres) * (normalized>-self.Thres)
        prunedx=x[Selected]
        
        Mean=torch.mean(prunedx)
        Var=torch.var(prunedx)
        self.Mean=self.alpha*self.Mean+(1.0-self.alpha)*Mean
        self.Var=self.alpha*self.Var+(1.0-self.alpha)*Var
        eps=1e-10
        bn= (self.gamma*(xorig-self.Mean)/torch.sqrt(self.Var+eps))+self.beta
        return bn
        
                
class BN1dRef(nn.Module):
    def __init__(self):
        super(BN1dRef , self).__init__()
        self.beta=nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.gamma=nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
    def forward(self, xorig):
        x=xorig.view(-1)
        Mean=torch.mean(x)
        Var=torch.var(x)
        eps=1e-10
        bn= (self.gamma*(xorig-Mean)/torch.sqrt(Var+eps))+self.beta
        return bn
        

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "SyncBN": NaiveSyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,  # keep for debugging
        }[norm]
    return norm(out_channels)


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class NaiveSyncBatchNorm(BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if comm.get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias
