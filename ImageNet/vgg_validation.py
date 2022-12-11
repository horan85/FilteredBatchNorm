import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import ImagenetLoaderValidation

class BN2dRef(nn.Module):
    def __init__(self, Channels ):
        super(BN2dRef , self).__init__()
        self.ChannelNum=Channels
        self.beta=Variable(torch.tensor([0.0]*Channels), requires_grad=True).cuda()
        self.gamma=Variable(torch.tensor([1.0]*Channels), requires_grad=True).cuda()
        self.beta=self.beta.reshape(1,Channels,1,1)
        self.gamma=self.gamma.reshape(1,Channels,1,1)
        
    def forward(self, xorig):
        x=xorig.permute([1,0,2,3])
        x=x.reshape((self.ChannelNum,-1))
        
        Mean=torch.mean(x, dim=-1).reshape((self.ChannelNum,1,1,1))
        Var=torch.var(x, dim=-1).reshape((self.ChannelNum,1,1,1))
        Mean=Mean.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        Var=Var.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        
        eps=1e-20
        beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        bn3= ((self.gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+self.beta
        #bn3= (((xorig-Mean))/(Var+eps)      )
        
        return bn3


class BN2d(nn.Module):
    def __init__(self, Channels, DropRate=0.02  ):
        super(BN2d , self).__init__()
        self.ChannelNum=Channels
        self.beta=Variable(torch.tensor([0.0]*Channels), requires_grad=True).cuda()
        self.gamma=Variable(torch.tensor([1.0]*Channels), requires_grad=True).cuda()
        self.beta=self.beta.reshape(1,Channels,1,1)
        self.gamma=self.gamma.reshape(1,Channels,1,1)
        self.DropRate=DropRate
        
    def forward(self, xorig):
        x=xorig.permute([1,0,2,3])
        x=x.reshape((self.ChannelNum,-1))
        
        Mean=torch.mean(x, dim=-1).reshape((self.ChannelNum,1,1,1))
        Var=torch.var(x, dim=-1).reshape((self.ChannelNum,1,1,1))
        Mean=Mean.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        Var=Var.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        
        eps=1e-10
        normalized= (xorig-Mean)/torch.sqrt(Var+eps)
        Thes=3.0
        Selected= ((normalized<Thes) * (normalized>-Thes)).float()
        #masked mean
        Mean=torch.sum(xorig*Selected, dim=[0,2,3])/torch.sum(Selected,dim=[0,2,3])
        Mean=Mean.reshape((1,self.ChannelNum,1,1))
        Mean=Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        Diff=(xorig - Mean)**2
        Var= torch.sum(Diff*Selected , dim=[0,2,3])/torch.sum(Selected,dim=[0,2,3])
        Var=Var.reshape((1,self.ChannelNum,1,1))
        Var=Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        

        eps=1e-20
        beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        bn3= ((self.gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+self.beta
        #bn3= (((xorig-Mean))/(Var+eps)      )
        
        return bn3


data_path='./imagenet/imagenet_tmp/raw_data/validation'

dataset_val= ImagenetLoaderValidation.ImageNetDataset(data_path, is_train = False)

BATCH_SIZE =16

data_loader_val = DataLoader(dataset_val, BATCH_SIZE, shuffle = True, num_workers=8)


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


   

def make_layers(cfg, batch_norm='None'):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm=='None':
                    layers += [conv2d, nn.ReLU(inplace=True)]                    
                elif batch_norm=="Normal":
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif batch_norm=="Ref":
                    layers += [conv2d, BN2dRef(v), nn.ReLU(inplace=True)]
                elif batch_norm=="Double":
                    layers += [conv2d, BN2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
batch_norm='Normal'
clf = VGG(make_layers(cfg_vgg16, batch_norm=batch_norm))        

PATH="vgg16_4_"+batch_norm+".pt"
clf.load_state_dict(torch.load(PATH))


clf.cuda()

SumAcc1=0.0
SumAcc5=0.0
Ind=0.0
clf.eval()
with torch.no_grad():
    for batch_id, batch in enumerate(data_loader_val):
            data=batch[0].cuda()
            label=batch[1].cuda()
            
            preds =clf(data)
            _, predind1 = preds.data.max(1)
            _, predind5 = torch.topk(preds.data,k=5, dim=1)

            acc1 = predind1.eq(label.data).float().mean().cpu() 
            print(acc1)
            label5=torch.unsqueeze(label.data,1)
            label5=label5.data.expand_as(predind5)
            correct5,_= predind5.eq(label5).max(1)

            acc5 = correct5.float().mean().cpu() 
            SumAcc1+=acc1.item()
            SumAcc5+=acc5.item()
            Ind+=1.0
print(SumAcc1/Ind)
print(SumAcc5/Ind)
     


