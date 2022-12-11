import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np



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
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../../mnist_data', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(), # first, convert image to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) ])), 
        batch_size=BatchSize, shuffle=True)

       
        if BNType=="Builtin":
                BNFunction=nn.BatchNorm1d   
        elif BNType=="Filtered": 
                BNFunction=BN1dFitlered
        elif BNType=="FilteredMoments": 
                BNFunction=BN1dFitleredMoments
        elif BNType=="Ref": 
                BNFunction=BN1dRef
        elif BNType=="None": 
                BNFunction=None
   

        
        class CNNClassifier(nn.Module):
            def __init__(self):
                super(CNNClassifier, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.dropout = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                if BNType=="Filtered" or BNType=="FilteredMoments":
                        self.bn3 = BNFunction(FilterValue)
                elif BNType=="Builtin":
                        self.bn3 = BNFunction(50)
                elif BNType=="None":
                        pass
                else:
                        self.bn3 = BNFunction()
                self.fc2 = nn.Linear(50, 10)
            
            def forward(self, x):
                x=F.max_pool2d(self.conv1(x), 2)
                
                x = F.relu(x)
                
                x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
                
                x = x.view(-1, 320)
                
               
                x=self.fc1(x)
                if BNType=="None":
                    x = F.relu(x)    
                else:
                        x = F.relu(self.bn3(x))
                x = F.dropout(x)
                
                x = self.fc2(x)
                
                return F.log_softmax(x)

        # create classifier and optimizer objects

        
        
        def train(epoch,Loss_history):
            
            clf.train()
            for batch_id, (data, label) in enumerate(train_loader):
                
                data=data.cuda()
                label=label.cuda()
                opt.zero_grad()
                preds = clf(data)
                loss = F.nll_loss(preds, label)
                loss.backward()
                opt.step()
                Loss_history.append(loss.item())
                #print(Grad_history)     

                predind = preds.data.max(1)[1] 
                acc = predind.eq(label.data).cpu().float().mean() 
 
                if batch_id % 100 == 0:
                    print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))
            return Loss_history
                
        Loss_history02=[]
        Loss_history01=[]
        Loss_history005=[]
        Loss_history001=[]
        clf = CNNClassifier()
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.02)
        opt = optim.SGD(clf.parameters(), lr=0.02, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history02=train(0,Loss_history02)
        
       
        torch.cuda.empty_cache()
        clf = CNNClassifier()
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.01)
        opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history01=train(0,Loss_history01)
        mincurve=np.minimum(Loss_history01,Loss_history02)
        maxcurve=np.maximum(Loss_history01,Loss_history02)
        
        torch.cuda.empty_cache()
        clf = CNNClassifier()
        clf.cuda()
        #opt = optim.Adam(clf.parameters(), lr=0.005)
        opt = optim.SGD(clf.parameters(), lr=0.005, momentum=0.5)
        for epoch in range(0, Epoch):
                Loss_history005=train(0,Loss_history005)
        mincurve=np.minimum(mincurve,Loss_history005)
        maxcurve=np.maximum(maxcurve,Loss_history005)
        
        torch.cuda.empty_cache()
        clf = CNNClassifier()
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

BatchSize=64
Epoch=3
Repeat=1
print("Test Builtin")
#RunTest("Builtin",Epoch,Repeat,BatchSize)
print("Test Filtered")
#RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
#RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
#RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
#RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,2.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,3.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,4.0)
#RunTest("FilteredMoments",Epoch,Repeat,BatchSize,5.0)
print("Test Ref")
RunTest("Ref",Epoch,Repeat,BatchSize)
print("Test None")
RunTest("None",Epoch,Repeat,BatchSize)






