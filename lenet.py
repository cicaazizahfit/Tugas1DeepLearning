!pip install torch torchvision

import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable as var 

n_batch = 64
learning_rate = 0.01
n_epoch = 3
n_print = 10

#### Part I : Loading Your Data 

T = torchvision.transforms.ToTensor()
train_data = torchvision.datasets.MNIST('mnist_data',train=True,download=True,transform=T)
val_data = torchvision.datasets.MNIST('mnist_data',train=False,download=True,transform=T)

train_dl = torch.utils.data.DataLoader(train_data,batch_size = n_batch)
val_dl = torch.utils.data.DataLoader(val_data,batch_size = n_batch)

#### Part II : Writing the Network
class myCNN(nn.Module):
  def __init__(self):
    super(myCNN,self).__init__()
    self.cnn1 = nn.Conv2d(1,3,3)
    self.cnn2 = nn.Conv2d(3,2,5)
    self.linear = nn.Linear(968,10)
    self.relu = nn.ReLU()
  
  def forward(self,x):
    n = x.size(0)
    x = self.relu(self.cnn1(x))
    x = self.relu(self.cnn2(x))
    x = x.view(n,-1)
    x = self.linear(x)
    return x

#### Part III : Writing the main Training loop

mycnn = myCNN().cuda()
cec = nn.CrossEntropyLoss()
optimizer = optim.Adam(mycnn.parameters(),lr = learning_rate)

def validate(model,data):
  # To get validation accuracy = (correct/total)*100.
  total = 0
  correct = 0
  for i,(images,labels) in enumerate(data):
    images = var(images.cuda())
    x = model(images)
    value,pred = torch.max(x,1)
    pred = pred.data.cpu()
    total += x.size(0)
    correct += torch.sum(pred == labels)
  return correct*100./total

for e in range(n_epoch):
  for i,(images,labels) in enumerate(train_dl):
    images = var(images.cuda())
    labels = var(labels.cuda())
    optimizer.zero_grad()
    pred = mycnn(images)
    loss = cec(pred,labels)
    loss.backward()
    optimizer.step()
    if (i+1) % n_print == 0:
      accuracy = float(validate(mycnn,val_dl))
      print('Epoch :',e+1,'Batch :',i+1,'Loss :',float(loss.data),'Accuracy :',accuracy,'%')

!nvidia-smi
