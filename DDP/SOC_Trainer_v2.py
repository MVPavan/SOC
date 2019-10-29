import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data_loader_spyder


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
train_dataset,test_dataset = data_loader_spyder.GetSOCdata(batch_len = 1000)

# Data loader
train_batch_size = len(train_dataset)/batch_len
test_batch_size = len(test_dataset)/batch_len

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=train_batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=test_batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
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

# Loss and optimizer
class SOC_Loss():
    def __init__(self,):
        pass

    def criterion(self,soc_est,soc_gt):
        max_sqer = (max(soc_est-soc_gt))**2
        return max_sqer+nn.MSELoss()(soc_est,soc_gt)

    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  

# Train the model
total_step = len(train_loader)
soc_loss = SOC_Loss()
for epoch in range(num_epochs):
    for i, (inputs, soc_gt) in enumerate(train_loader):  
        # Move tensors to the configured device
        inputs = inputs.to(device)
        soc_gt = soc_gt.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = soc_loss.criterion(outputs, soc_gt)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, soc_gt in test_loader:
        inputs = inputs.reshape(-1, 28*28).to(device)
        soc_gt = soc_gt.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += soc_gt.size(0)
        correct += (predicted == soc_gt).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')