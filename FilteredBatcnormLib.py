import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np



class BN2dReimplemented(nn.Module):
    def __init__(self,NumFeatures):
        super(BN2dReimplemented , self).__init__()
        self.NumFeatures=NumFeatures
        self.beta=nn.Parameter(torch.zeros(NumFeatures), requires_grad=True)
        self.gamma=nn.Parameter(torch.ones(NumFeatures), requires_grad=True)
        self.alpha=0.9
        self.Mean=torch.zeros(NumFeatures).cuda()
        self.Var=torch.ones(NumFeatures).cuda()

    def SetThreshold(Thres):
        self.Thres=Thres 
             
    def forward(self, xorig):        
        x=xorig.permute([0,2,3,1])
        x=x.reshape((-1,self.NumFeatures))
        eps=1e-5
        if self.training:
                Mean=torch.mean(x,dim=0)
                Var=torch.var(x,dim=0,unbiased=True)
                Mean4D=Mean.reshape((1,self.NumFeatures,1,1))
                Var4D=Var.reshape((1,self.NumFeatures,1,1))
                beta4D=self.beta.reshape((1,self.NumFeatures,1,1))
                gamma4D=self.gamma.reshape((1,self.NumFeatures,1,1))
                Mean4D=Mean4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
                Var4D=Var4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
                beta4D=beta4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
                gamma4D=gamma4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           
                bn= (gamma4D*(xorig-Mean4D)/torch.sqrt(Var4D+eps))+beta4D          
                self.Mean= (self.alpha*self.Mean+(1.0-self.alpha)*Mean).detach()
                self.Var=  (self.alpha*self.Var+(1.0-self.alpha)*Var).detach()
        else:
           Mean4D=self.Mean.reshape((1,self.NumFeatures,1,1))
           Var4D=self.Var.reshape((1,self.NumFeatures,1,1))
           beta4D=self.beta.reshape((1,self.NumFeatures,1,1))
           gamma4D=self.gamma.reshape((1,self.NumFeatures,1,1))
           Mean4D=Mean4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           Var4D=Var4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           beta4D=beta4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           gamma4D=gamma4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           
           bn= (gamma4D*(xorig-Mean4D)/torch.sqrt(Var4D+eps))+beta4D          
        return bn
 

 
class BN1dReimplemented(nn.Module):
    def __init__(self,NumFeatures):
        super(BN1dReimplemented , self).__init__()
        self.NumFeatures=NumFeatures
        self.beta=nn.Parameter(torch.zeros(NumFeatures), requires_grad=True)
        self.gamma=nn.Parameter(torch.ones(NumFeatures), requires_grad=True)
        self.alpha=0.9
        self.Mean=torch.zeros(NumFeatures).cuda()
        self.Var=torch.ones(NumFeatures).cuda()

    def SetThreshold(Thres):
        self.Thres=Thres 
             
    def forward(self, xorig):        
        x=xorig
        eps=1e-5
        if self.training:
                Mean=torch.mean(x,dim=0)
                Var=torch.var(x,dim=0,unbiased=True)
                bn= (self.gamma*(xorig-Mean)/torch.sqrt(Var+eps))+self.beta  
                self.Mean= (self.alpha*self.Mean+(1.0-self.alpha)*Mean).detach()
                self.Var=  (self.alpha*self.Var+(1.0-self.alpha)*Var).detach()
        else:
           bn= (self.gamma*(xorig-self.Mean)/torch.sqrt(self.Var+eps))+self.beta              
        return bn
 
class BN1dFitleredMoments(nn.Module):
    def __init__(self,NumFeatures,Thres=1.0):
        super(BN1dFitleredMoments , self).__init__()
        self.Thres=Thres
        self.NumFeatures=NumFeatures
        self.beta=nn.Parameter(torch.zeros(NumFeatures), requires_grad=True)
        self.gamma=nn.Parameter(torch.ones(NumFeatures), requires_grad=True)
        self.alpha=0.9
        self.Mean=torch.zeros(NumFeatures).cuda()
        self.Var=torch.ones(NumFeatures).cuda()

    def SetThreshold(Thres):
        self.Thres=Thres 
             
    def forward(self, xorig):        
        x=xorig
        eps=1e-5
        if self.training:
                Mean=torch.mean(x,dim=0)
                Var=torch.var(x,dim=0,unbiased=True)
                
                normalized= (x-Mean)/torch.sqrt(Var+eps)
                Selected=(normalized<self.Thres) * (normalized>-self.Thres)
                #print(Selected.shape)
                #prunedx=x*Selected
                FilteredMean= torch.sum(x*Selected,dim=0)/torch.sum(Selected,dim=0)
                FilteredVar=  torch.sum( ((x-FilteredMean)*Selected)*((x-FilteredMean)*Selected),dim=0 ) /(torch.sum(Selected,dim=0)-1)
                bn= (self.gamma*(xorig-FilteredMean)/torch.sqrt(FilteredVar+eps))+self.beta  
                self.Mean= (self.alpha*self.Mean+(1.0-self.alpha)*FilteredMean).detach()
                self.Var=  (self.alpha*self.Var+(1.0-self.alpha)*FilteredVar).detach()
        else:
           bn= (self.gamma*(xorig-self.Mean)/torch.sqrt(self.Var+eps))+self.beta              
        return bn


class BN2dFitleredMoments(nn.Module):
    def __init__(self,NumFeatures,Thres=1.0):
        super(BN2dFitleredMoments , self).__init__()
        self.NumFeatures=NumFeatures
        self.Thres=Thres 
        self.beta=nn.Parameter(torch.zeros(NumFeatures), requires_grad=True)
        self.gamma=nn.Parameter(torch.ones(NumFeatures), requires_grad=True)
        self.alpha=0.9
        self.Mean=torch.zeros(NumFeatures).cuda()
        self.Var=torch.ones(NumFeatures).cuda()

    def SetThreshold(Thres):
        self.Thres=Thres 
             
    def forward(self, xorig):        
        x=xorig.permute([0,2,3,1])
        x=x.reshape((-1,self.NumFeatures))
        eps=1e-5
        if self.training:
                Mean=torch.mean(x,dim=0)
                Var=torch.var(x,dim=0,unbiased=True)
                
                normalized= (x-Mean)/torch.sqrt(Var+eps)
                Selected=(normalized<self.Thres) * (normalized>-self.Thres)
                #print(Selected.shape)
                #prunedx=x*Selected
                FilteredMean= torch.sum(x*Selected,dim=0)/torch.sum(Selected,dim=0)
                FilteredVar=  torch.sum( ((x-FilteredMean)*Selected)*((x-FilteredMean)*Selected),dim=0 ) /(torch.sum(Selected,dim=0)-1)
                
                Mean4D=FilteredMean.reshape((1,self.NumFeatures,1,1))
                Var4D=FilteredVar.reshape((1,self.NumFeatures,1,1))
                beta4D=self.beta.reshape((1,self.NumFeatures,1,1))
                gamma4D=self.gamma.reshape((1,self.NumFeatures,1,1))
                Mean4D=Mean4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
                Var4D=Var4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
                beta4D=beta4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
                gamma4D=gamma4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           
                bn= (gamma4D*(xorig-Mean4D)/torch.sqrt(Var4D+eps))+beta4D          
                self.Mean= (self.alpha*self.Mean+(1.0-self.alpha)*FilteredMean).detach()
                self.Var=  (self.alpha*self.Var+(1.0-self.alpha)*FilteredVar).detach()
        else:
           Mean4D=self.Mean.reshape((1,self.NumFeatures,1,1))
           Var4D=self.Var.reshape((1,self.NumFeatures,1,1))
           beta4D=self.beta.reshape((1,self.NumFeatures,1,1))
           gamma4D=self.gamma.reshape((1,self.NumFeatures,1,1))
           Mean4D=Mean4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           Var4D=Var4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           beta4D=beta4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           gamma4D=gamma4D.expand((xorig.shape[0],self.NumFeatures,xorig.shape[2],xorig.shape[3]))
           
           bn= (gamma4D*(xorig-Mean4D)/torch.sqrt(Var4D+eps))+beta4D          
        return bn
  
  
