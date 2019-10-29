import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data_loader_spyder
import random

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




if torch.cuda.is_available() :
    CUDA_VISIBLE_DEVICES=3
    device = torch.device('cuda:3')
else:
    torch.device('cpu')


torch.cuda.current_device()
# Hyper-parameters 
input_size = 4
hidden_size = 4
num_classes = 1
num_epochs = 5000
batch_len = 1000
learning_rate = 0.0001

# MNIST dataset 
train_dataset,test_dataset = data_loader_spyder.GetSOCdata(batch_len = 1000, pkl = False)

# Data loader
# train_batch_size = int(len(train_dataset)/batch_len)
# test_batch_size = int(len(test_dataset)/batch_len)
train_batch_size = batch_len
test_batch_size = batch_len


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=train_batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=test_batch_size, 
                                          shuffle=False)
# Fully connected neural network with one hidden layer
# Loss and optimizer
class SOC_Loss():
    def __init__(self,):
        pass

    def criterion(self,soc_est,soc_gt):
#         soc_gt=soc_gt/100
        err = soc_est-soc_gt
        err = torch.max(err)
#         print(err.type())
        max_sqer = err**2
        mse_err = nn.MSELoss()(soc_est,soc_gt)
#         print(max_sqer.item(),mse_err.item())
        return max_sqer+mse_err
    
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  
# Train the model
total_step = len(train_loader)
soc_loss = SOC_Loss()
for epoch in range(num_epochs):
    for i, (inputs, soc_gt) in enumerate(train_loader):
        
        inputs = inputs.float().to(device)
        soc_gt = soc_gt.float().to(device)
#         print(inputs.type(), soc_gt.type())
        # Move tensors to the configured device
#         print(i,[len(inputs),len(inputs[0]),len(inputs[1]),len(inputs[2]),len(inputs[3])],len(soc_gt))
#         inputs = inputs.to(device)
#         soc_gt = soc_gt.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = soc_loss.criterion(outputs, soc_gt)        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            k=random.randint(1,batch_len)
            print(outputs.data[k],soc_gt.data[k])
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for inputs, soc_gt in test_loader:
#         inputs = inputs.float().to(device)
#         soc_gt = soc_gt.float().to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += soc_gt.size(0)
#         correct += (predicted == soc_gt).sum().item()

#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')