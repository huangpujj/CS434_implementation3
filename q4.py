import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
sns.set()

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

learn_rate = float(sys.argv[1])
batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=datasets.CIFAR10('./', train=True,download=True,transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])),
    batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(
    dataset=datasets.CIFAR10('./', train=False,transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                   ])),
    batch_size=batch_size, shuffle=False)

train_batch = []
validate_batch = []
for i, val in enumerate(train_loader):
    if i < (40000 / batch_size):
        train_batch.append(val)
    else:
        validate_batch.append(val)
# for (X_train, y_train) in train_loader:
#     print('X_train:', X_train.size(), 'type:', X_train.type())
#     print('y_train:', y_train.size(), 'type:', y_train.type())
#     break
# pltsize=1
# plt.figure(figsize=(10*pltsize, pltsize))

# for i in range(10):
#     plt.subplot(1,10,i+1)
#     plt.axis('off')
#     plt.imshow(X_train[i,:,:,:].numpy().reshape(32,32,3))
#     plt.title('Class: '+str(y_train[i].item()))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.sigmoid(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.sigmoid(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

model = Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.5)
criterion = nn.CrossEntropyLoss()

print(model)




def train(epoch, log_interval=50):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_batch):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_batch)*batch_size,
                100. * batch_idx / len(train_batch), loss.data.item()))

def validate(loss_vector, accuracy_vector,batch):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in batch:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(batch)
    loss_vector.append(val_loss)

    accuracy = correct.to(torch.float32) / len(batch*batch_size)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(batch)*batch_size,100 * accuracy))

epochs = 50

lossv, accv = [], []
losst, acct = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv,accv,validate_batch)
    validate(losst, acct,train_batch)

# plt.figure(figsize=(5,3))
# plt.plot(np.arange(1,epochs+1), lossv)
# plt.title('validation loss')

# plt.figure(figsize=(5,3))
# plt.plot(np.arange(1,epochs+1), accv)
# plt.title('validation accuracy');

epoch_label = [i for i in range(1, epochs + 1)]
plt.title('Training accuracy and Validation accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(epoch_label, acct, marker='o')
plt.plot(epoch_label, accv, marker='o')
plt.legend(['Training accuracy', 'Validation accuracy'])
plt.savefig('q4.png')
plt.clf()