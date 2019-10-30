import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data_loader_spyder
import random
import pickle
import Pytorch_Utils


class SOC_Loss():
    def __init__(self,):
        self.scale = 100

    def criterion(self,soc_est,soc_gt):
        err = soc_est-soc_gt
        err = torch.max(err)
        print(torch.max(err))
        # max_sqer = err**2
        max_abe = abs(err)
        mae_err = nn.L1Loss()(soc_est,soc_gt)        
        mse_err = nn.MSELoss()(soc_est,soc_gt)
        closs = 0*max_abe+0*mae_err+self.scale*mse_err
        # print(max_abe.item(),mae_err.item())
        # print(mae_err.item(),max_sqer.item(),mse_err.item())
        return closs
 
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
        # out = torch.clamp(out, min=0, max=1)
        # out = self.relu(out)
        return out

class ModelClass():
    def __init__(self,):
        self.myUtils = Pytorch_Utils.MyUtils()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() :
        #     CUDA_VISIBLE_DEVICES=4
            self.device = torch.device('cuda:4')
        else:
            self.device = torch.device('cpu')
        # torch.cuda.current_device()
        self.HyperParams()
        self.DataProcess()

    def HyperParams(self,):# Hyper-parameters 
        self.input_size = 4
        self.hidden_size = 8
        self.num_classes = 1
        self.num_epochs = 1
        self.batch_len = 1
        self.learning_rate = 0.0001
        self.train_batch_size = self.batch_len
        self.test_batch_size = 1000

    def DataProcess(self,):
        train_dataset,test_dataset = data_loader_spyder.GetSOCdata(self.batch_len , pkl = False)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=self.test_batch_size, 
                                                shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=self.test_batch_size, 
                                                shuffle=False)

    def ModelInit(self,):
        self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=False)
        self.soc_loss = SOC_Loss()

    def trainer(self,):
        total_step = len(self.train_loader)
        total_loss = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            if epoch > 5:
                self.soc_loss.scale = self.soc_loss.scale*10

            for i, (inputs, soc_gt) in enumerate(self.train_loader):
                if i>0:
                    break
                inputs = inputs.float().to(self.device)
                soc_gt = soc_gt.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.soc_loss.criterion(outputs, soc_gt)        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss = epoch_loss + loss.item()
                total_loss = total_loss + loss.item()
                if (i+1) % 100 == 0:
                    if self.batch_len>1:
                        k=random.randint(1,self.batch_len-1)
                    else:
                        k = 0
                    print(k,outputs.data[k][0],soc_gt.data[k])
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
                    self.myUtils.writer("ModelLoss",{ 'EpochAvgLoss': epoch_loss/(i+1),
                                                        'TotalAvgLoss': total_loss/(epoch*self.batch_len+i+1)},(epoch*self.batch_len+i+1))
            
            self.myUtils.checkpoint_saver(self.model,self.optimizer,epoch,self.batch_len,total_loss/(epoch*self.batch_len),"SOC_481_NN")
            self.Tester(epoch)
        self.myUtils.model_state_dict_saver(self.model,"SOC_481",self.num_epochs)

    def Tester(self,epoch=1):
        with torch.no_grad():
            total_test_error = 0
            sample_length = len(self.test_loader)/self.test_batch_size
            for i, (inputs, soc_gt) in enumerate(self.train_loader):
                inputs = inputs.float().to(self.device)
                soc_gt = soc_gt.float().to(self.device)
                outputs = self.model(inputs)
                predicted = torch.max(outputs.data, 1)
                total_test_error = abs(torch.mean(predicted-soc_gt).item())+total_test_error
            self.myUtils.writer("Average Error",total_test_error/sample_length,epoch)
        print('Average error of the network till epoch: {} is {}%'.format(epoch,total_test_error/sample_length))


if __name__ == "__main__":

    self = ModelClass()
