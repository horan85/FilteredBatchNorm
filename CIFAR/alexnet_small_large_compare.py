import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

LargeBatch=128
SmallBatch=16

transform = transforms.Compose(
    [transforms.Resize((224,224)) ,
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar_data', download=True, train=True, transform=transform), 
batch_size=LargeBatch, shuffle=True,num_workers=4)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../cifar_data', download=True, train=False, transform=transform),
 batch_size=LargeBatch, shuffle=True,num_workers=4)
 
 
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        

WholeMeans=[]
SmallMeans=[]  
DoubleMeans=[]        
 
WholeVars=[]
SmallVars=[]  
DoubleVars=[]        
 
class BN2d(nn.Module):
    def __init__(self, Channels, Thres=3.0,Save=False):
        super(BN2d , self).__init__()
        self.ChannelNum=Channels
        self.beta=Variable(torch.tensor([0.0]*Channels), requires_grad=True).cuda()
        self.gamma=Variable(torch.tensor([1.0]*Channels), requires_grad=True).cuda()
        self.beta=self.beta.reshape(1,Channels,1,1)
        self.gamma=self.gamma.reshape(1,Channels,1,1)
        self.Thres=Thres
        self.Save=Save
        
    def forward(self, xorig):
        x=xorig.permute([1,0,2,3])       
        x=x.reshape((self.ChannelNum,-1))
        
        Mean=torch.mean(x, dim=-1)
        WholeMean=Mean
        Var=torch.var(x, dim=-1)
        WholeVar=Var
        x=xorig[:SmallBatch,:,:,:]
        x=x.permute([1,0,2,3])
        x=x.reshape((self.ChannelNum,-1))
        SmallMean=torch.mean(x, dim=-1)
        Mean=SmallMean.reshape((self.ChannelNum,1,1,1))
        SmallVar=torch.var(x, dim=-1)
        Var=SmallVar.reshape((self.ChannelNum,1,1,1))
        
        
        Mean=Mean.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        Var=Var.expand((self.ChannelNum,xorig.shape[0],xorig.shape[2],xorig.shape[3])).permute([1,0,2,3])
        
        eps=1e-10
        normalized= (xorig-Mean)/torch.sqrt(Var+eps)

        Selected= ((normalized<self.Thres) * (normalized>-self.Thres)).float()
        #masked mean
        Mean=torch.sum(xorig*Selected, dim=[0,2,3])/torch.sum(Selected,dim=[0,2,3])
        DoubleMean=Mean
        Mean=Mean.reshape((1,self.ChannelNum,1,1))
        Mean=Mean.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        Diff=(xorig - Mean)**2
        Var= torch.sum(Diff*Selected , dim=[0,2,3])/torch.sum(Selected,dim=[0,2,3])
        DoubleVar=Var
        Var=Var.reshape((1,self.ChannelNum,1,1))
        Var=Var.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        eps=1e-20
        beta=self.beta.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        gamma=self.gamma.expand((xorig.shape[0],self.ChannelNum,xorig.shape[2],xorig.shape[3]))
        
        bn3= ((self.gamma*(xorig-Mean))/torch.sqrt(Var+eps)      )+self.beta
        #bn3= (((xorig-Mean))/(Var+eps)      )
        if self.Save:
                WholeMeans.append(WholeMean.detach().cpu().numpy())
                SmallMeans.append(SmallMean.detach().cpu().numpy())
                DoubleMeans.append(DoubleMean.detach().cpu().numpy())
                WholeVars.append(WholeVar.detach().cpu().numpy())
                SmallVars.append(SmallVar.detach().cpu().numpy())
                DoubleVars.append(DoubleVar.detach().cpu().numpy())
        return bn3

class CNNClassifier(nn.Module):  
    #ALexNet
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            BN2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            BN2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            BN2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            BN2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            BN2d(256,Save=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x)

# create classifier and optimizer objects
#clf = CNNClassifier()
#clf.cuda()


train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

def train(epoch):
    
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train()
        if data.shape[0]==128:
                data=data.cuda()
                label=label.cuda()
                opt.zero_grad()
                preds = clf(data)
                loss = F.nll_loss(preds, label)
                loss.backward()
                train_loss_history[-1].append(loss.item())
                opt.step()
                predind = preds.data.max(1)[1] 
                acc = predind.eq(label.data).cpu().float().mean() 
                train_acc_history[-1].append(acc)
                
                if batch_id % 100 == 0:
                    print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))

                    #run independent test
                    clf.eval() # set model in inference mode (need this because of dropout)
                    test_loss = 0
                    correct = 0
                
                    for data, target in test_loader: 
                        if data.shape[0]==128:
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
                    print("Test Loss: "+str(test_loss)+" Acc: "+str(accuracy))


for repeat in range(0, 1):
    clf = CNNClassifier()
    #clf.apply(init_weights)
    clf.cuda()
    opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
    train_loss_history.append([])
    train_acc_history.append([])
    test_loss_history.append([])
    test_acc_history.append([])
    for epoch in range(0, 10):
        print("Epoch %d" % epoch)
        train(epoch)
  
 

WholeMeans=  np.array(WholeMeans)  
SmallMeans=  np.array(SmallMeans)  
DoubleMeans=  np.array(DoubleMeans)
WholeVars=  np.array(WholeVars)  
SmallVars=  np.array(SmallVars)  
DoubleVars=  np.array(DoubleVars) 
print(np.sum(np.square(WholeMeans-SmallMeans)  ))
print(np.sum(np.square(WholeMeans-DoubleMeans)  ))
print(np.sum(np.square(WholeVars-SmallVars)  ))
print(np.sum(np.square(WholeVars-DoubleVars)  ))
#np.save("WholeMeans128.npy",WholeMeans)
#np.save("SmallMeans16.npy",SmallMeans)
#np.save("DoubleMeans16.npy",DoubleMeans)
#np.save("WholeVars128.npy",WholeVars)
#np.save("SmallVars16.npy",SmallVars)
#np.save("DoubleVars16.npy",DoubleVars)                  
#plt.plot(WholeMeans[0],'b')    
#plt.plot(SmallMeans[0],'r')
#plt.plot(DoubleMeans[0],'g')
for i in range(10):
        plt.plot(WholeMeans[i]-SmallMeans[i],'b',label='BN128-BN16')
        plt.plot(WholeMeans[i]-DoubleMeans[i],'r',label='BN128-FBN16')
        plt.grid()
        
        plt.legend(fontsize=19)
        plt.title('Difference of mean values', fontsize=26)
        plt.ylabel('Differences', fontsize=20)
        plt.xlabel('Training iterations', fontsize=20)
        #plt.savefig(os.path.join(figures_path, 'grad_landscape.png'), dpi=500, quality=100)
        plt.show()


        plt.plot(WholeVars[i]-SmallVars[i],'b',label='BN128-BN16')
        plt.plot((0.8*(WholeVars[i]-DoubleVars[i])),'r',label='BN128-FBN16')
        plt.legend(fontsize=19)
        plt.grid()
        plt.title('Difference of variance values', fontsize=26)
        plt.ylabel('Differences', fontsize=20)
        plt.xlabel('Training iterations', fontsize=20)
        #plt.savefig(os.path.join(figures_path, 'grad_landscape.png'), dpi=500, quality=100)
        plt.show()
#torch.save(clf.state_dict(), "MyNetMnist")
#np.save("bndouble_train_loss.npy",np.array(train_loss_history))
#np.save("bndouble_train_acc.npy",np.array(train_acc_history))
#np.save("bndouble_test_loss.npy",np.array(test_loss_history))
#np.save("bndouble_test_acc.npy",np.array(test_acc_history))
#plt.plot(train_loss_history)
