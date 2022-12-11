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
import ImagenetLoader
import time



class BN2dRef(nn.Module):
    def __init__(self, Channels ):
        super(BN2dRef , self).__init__()
        self.ChannelNum=Channels
        self.beta=nn.Parameter(torch.tensor([0.0]*int(Channels)).reshape(1,Channels,1,1), requires_grad=True)
        self.gamma=nn.Parameter(torch.tensor([1.0]*int(Channels)).reshape(1,Channels,1,1), requires_grad=True)
                
        
    def forward(self, xorig):
                x=xorig.permute([1,0,2,3])
                x=x.reshape((self.ChannelNum,-1))
                
                Mean=torch.mean(x, dim=-1).reshape((1,self.ChannelNum,1,1))
                Var=torch.var(x, dim=-1).reshape((1,self.ChannelNum,1,1))
                Var=Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                Mean=Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                eps=1e-10
                beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
                
                bnfiltered= ((gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+beta
                return bnfiltered

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
        

def RunTest(BNType,Epoch,Repeat, BatchSize, FilterValue=4.0):
        # download and transform train dataset
               
        data_path='./imagenet/imagenet_tmp/raw_data/train'
        dataset_train, dataset_test = ImagenetLoader.get_imagenet_datasets(data_path)


        BATCH_SIZE =16

        train_loader = DataLoader(dataset_train, BATCH_SIZE, shuffle = True, num_workers=8)

       
        


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
                        elif batch_norm=="Builtin":
                            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                        elif batch_norm=="Ref":
                            layers += [conv2d, BN2dRef(v), nn.ReLU(inplace=True)]
                        elif batch_norm=="Filtered":
                            layers += [conv2d, BN2dFitlered(v), nn.ReLU(inplace=True)]
                        in_channels = v
                return nn.Sequential(*layers)

        
               

        
        def train(epoch,Loss_history):
            clf.train()
            criterion = nn.CrossEntropyLoss()
            for batch_id, batch in enumerate(train_loader):
                if batch_id<10000:
                    data=batch[0].cuda()
                    label=batch[1].cuda()
                    opt.zero_grad()
                    preds =clf(data)
                    loss = criterion(preds, label)
                    if not torch.isnan(loss).item() and loss.item()<1000:
                        loss.backward()
                        opt.step()
                        Loss_history.append(loss.item())
                   
                    if batch_id % 100 == 0:
                        print("Train Loss: "+str(loss.item())) 
                else:
                    return Loss_history
                
            return Loss_history
        cfg_vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']        
        Loss_history02=[]
        Loss_history01=[]
        Loss_history005=[]
        Loss_history001=[]
        clf = VGG(make_layers(cfg_vgg16, batch_norm=BNType)) 
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.02)
        opt = optim.SGD(clf.parameters(), lr=0.02, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history02=train(0,Loss_history02)
        
       
        torch.cuda.empty_cache()
        clf = VGG(make_layers(cfg_vgg16, batch_norm=BNType)) 
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.01)
        opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history01=train(0,Loss_history01)
        mincurve=np.minimum(Loss_history01,Loss_history02)
        maxcurve=np.maximum(Loss_history01,Loss_history02)
        
        torch.cuda.empty_cache()
        clf = VGG(make_layers(cfg_vgg16, batch_norm=BNType)) 
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.005)
        opt = optim.SGD(clf.parameters(), lr=0.005, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history005=train(0,Loss_history005)
        mincurve=np.minimum(mincurve,Loss_history005)
        maxcurve=np.maximum(maxcurve,Loss_history005)
        
        torch.cuda.empty_cache()
        clf = VGG(make_layers(cfg_vgg16, batch_norm=BNType)) 
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.001)
        opt = optim.SGD(clf.parameters(), lr=0.001, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history001=train(0,Loss_history001)
        torch.cuda.empty_cache()
        mincurve=np.minimum(mincurve,Loss_history001)
        maxcurve=np.maximum(maxcurve,Loss_history001)
        if BNType=="Filtered" or BNType=="FilteredMoments": 
                np.save("grad/"+BNType+"_"+str(BatchSize)+"_"+str(FilterValue)+"_minloss.npy",np.array(mincurve))
                np.save("grad/"+BNType+"_"+str(BatchSize)+"_"+str(FilterValue)+"_maxloss.npy",np.array(maxcurve))
        else: 
                np.save("grad/"+BNType+"_"+str(BatchSize)+"_minloss.npy",np.array(mincurve))
                np.save("grad/"+BNType+"_"+str(BatchSize)+"_maxloss.npy",np.array(maxcurve))

BatchSize=16
Epoch=1
Repeat=1
print("Test Builtin")
RunTest("Builtin",Epoch,Repeat,BatchSize)
print("Test Filtered")
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
#RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
#RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,2.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,3.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,4.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,5.0)
print("Test Ref")
RunTest("Ref",Epoch,Repeat,BatchSize)
print("Test None")
RunTest("None",Epoch,Repeat,BatchSize)





