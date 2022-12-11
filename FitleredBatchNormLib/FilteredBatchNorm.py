import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np



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
        

