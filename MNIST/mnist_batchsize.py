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

        # download and transform test dataset
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../../mnist_data', download=True, train=False,transform=transforms.Compose([
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

        train_loss_history = []
        train_acc_history = []
        test_loss_history = []
        test_acc_history = []

        def train(epoch,TrainInd):
            
            for batch_id, (data, label) in enumerate(train_loader):
                TrainInd+=1
                clf.train()
                data=data.cuda()
                label=label.cuda()
                opt.zero_grad()
                preds = clf(data)
                loss = F.nll_loss(preds, label)
                loss.backward(retain_graph=True)
                train_loss_history[-1].append(loss.item())
                opt.step()
                predind = preds.data.max(1)[1] 
                acc = predind.eq(label.data).cpu().float().mean() 
                train_acc_history[-1].append(acc)
                
                if TrainInd % 100 == 0:
                    #print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))

                    #run independent test
                    clf.eval() # set model in inference mode (need this because of dropout)
                    test_loss = 0
                    correct = 0
                
                    for data, target in test_loader: 
                        data=data.cuda()
                        target=target.cuda()  
                        with torch.no_grad():    
                           output = clf(data)
                           test_loss += F.nll_loss(output, target).item()
                           pred = output.data.max(1)[1] 
                           correct += pred.eq(target.data).cpu().sum()

                    test_loss = test_loss
                    test_loss /= len(test_loader) # loss function already averages over batch size
                    accuracy =  correct.item() / len(test_loader.dataset)
                    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    #    test_loss, correct, len(test_loader.dataset),
                    #    accuracy))
                    test_acc_history[-1].append(accuracy)
                    test_loss_history[-1].append(test_loss)
                    #print("Test Loss: "+str(test_loss)+" Acc: "+str(accuracy))
            return TrainInd

        for repeat in range(0, Repeat):
            print("repeat: "+str(repeat))
            clf = CNNClassifier()
            clf.cuda()
            #opt = optim.Adam(clf.parameters(), lr=0.01)
            opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
            train_loss_history.append([])
            train_acc_history.append([])
            test_loss_history.append([])
            test_acc_history.append([])
            TrainInd=0
            epoch=0
            while TrainInd<5000:
                TrainInd=train(epoch,TrainInd)
            torch.cuda.empty_cache()
        if BNType=="Filtered" or BNType=="FilteredMoments": 
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_"+str(FilterValue)+"_train_loss.npy",np.array(train_loss_history))
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_"+str(FilterValue)+"_train_acc.npy",np.array(train_acc_history))
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_"+str(FilterValue)+"_test_loss.npy",np.array(test_loss_history))
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_"+str(FilterValue)+"_test_acc.npy",np.array(test_acc_history))

        else: 
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_train_loss.npy",np.array(train_loss_history))
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_train_acc.npy",np.array(train_acc_history))
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_test_loss.npy",np.array(test_loss_history))
                np.save("resbatch/"+BNType+"_"+str(BatchSize)+"_test_acc.npy",np.array(test_acc_history))

Epoch=3
Repeat=10
BatchSize=256
print("Test Builtin 256" )
RunTest("Builtin",Epoch,Repeat,BatchSize)
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
BatchSize=128
print("Test Builtin 128" )
RunTest("Builtin",Epoch,Repeat,BatchSize)
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
print("Test Builtin 64" )
BatchSize=64
RunTest("Builtin",Epoch,Repeat,BatchSize)
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
print("Test Builtin 32" )
BatchSize=32
RunTest("Builtin",Epoch,Repeat,BatchSize)
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
print("Test Builtin 16" )
BatchSize=16
RunTest("Builtin",Epoch,Repeat,BatchSize)
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
print("Test Builtin 8" )
BatchSize=8
RunTest("Builtin",Epoch,Repeat,BatchSize)
RunTest("Filtered",Epoch,Repeat,BatchSize,2.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,3.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,4.0)
RunTest("Filtered",Epoch,Repeat,BatchSize,5.0)
