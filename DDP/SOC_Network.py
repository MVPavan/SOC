import torch
import torch.nn as nn
import torch.utils.data
import SOC_Data_v2 as soc_data
import random
import pickle
import Pytorch_Utils

scale3 = 10
scale2 = 0
scale1 = 0
class SOC_Loss():
    def __init__(self,):
        self.scale1 = 0
        self.scale2 = 0
        self.scale3 = scale3
        self.mse_loss = nn.MSELoss()

    def criterion(self,soc_est,soc_gt):
        # err = soc_est*100-soc_gt*100
        # err = torch.max(err)
        # print(err)
        # max_sqer = err**2
        # max_abe = abs(err)
        # mae_err = nn.L1Loss()(soc_est,soc_gt)        
        mse_err = self.mse_loss(soc_est,soc_gt)
        # closs = self.scale1*max_abe#.item()+self.scale2*mae_err.item()+self.scale3*mse_err.item()
        closs = self.scale3*mse_err
        # print(max_abe.item(),mae_err.item())
        # print(max_abe.item())#,max_sqer.item(),mse_err.item())
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
        return out

class ModelClass():
    def __init__(self,):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available() :
        #     CUDA_VISIBLE_DEVICES=4
            self.device = torch.device('cuda:0')
            print("GPU Available !!!")
        else:
            self.device = torch.device('cpu')
        # torch.cuda.current_device()
        self.HyperParams()
        self.DataProcess()
        self.ModelInit()

    def HyperParams(self,):# Hyper-parameters 
        [self.input_size, self.hidden_size, self.num_classes] = [4, 8, 1]
        self.num_epochs = 15
        self.batch_len = 1
        self.learning_rate = 0.0001
        self.train_batch_size = self.batch_len
        self.test_batch_size = 1000
        save_str = "Exp_{}{}{}_Epochs_{}_bl_{}_lr_{}_ls1_{}_ls2_{}_ls3_{}".format(
            self.input_size, self.hidden_size, self.num_classes,
            self.num_epochs,self.batch_len,self.learning_rate,scale1, scale2, scale3
        )
        self.myUtils = Pytorch_Utils.MyUtils(save_path_dir=save_str)
        

    def DataProcess(self,):
        train_dataset,test_dataset = soc_data.GetSOCdata(self.batch_len , pkl = False)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=self.train_batch_size, 
                                                shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=self.test_batch_size, 
                                                shuffle=False)

    def ModelInit(self,):
        self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=False)
        self.soc_loss = SOC_Loss()

    # def ModelLoad(self,model_path)
    def Trainer(self,):
        total_step = len(self.train_loader)
        for epoch in range(self.num_epochs):
            batch_loss = 0
            batch_loss_update = 0
            if epoch > 8:
                [self.soc_loss.scale1,self.soc_loss.scale2,self.soc_loss.scale3] = [self.soc_loss.scale1,self.soc_loss.scale2,self.soc_loss.scale3]

            for i, (inputs, soc_gt) in enumerate(self.train_loader):
                # if i>10000:
                #     break
                inputs = inputs.float().to(self.device)
                soc_gt = soc_gt.float().to(self.device)
                outputs = self.model(inputs)
                loss = self.soc_loss.criterion(outputs, soc_gt)        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_loss = batch_loss + loss.item()
                if (i+1) % 300 == 0:
                    batch_loss_update = batch_loss/300
                    batch_loss = 0
                    if self.batch_len>1:
                        k=random.randint(1,self.batch_len-1)
                    else:
                        k = 0
                    print(k,outputs.data[k][0],soc_gt.data[k])
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Batch_Loss : {:.4f}'.format(epoch+1, self.num_epochs, i+1, total_step, loss.item(),batch_loss_update))
                    k1 = epoch*(total_step/self.batch_len)+i+1
                    self.myUtils.writer.add_scalars("ModelLoss",{'Loss': loss.item()*100,
                                                                 'BatchAvgLoss': batch_loss_update*100,
                                                                },k1)
            
            self.myUtils.checkpoint_saver(self.model,self.optimizer,epoch,self.batch_len,batch_loss_update,"SOC_481_NN")
            self.Tester(epoch)
        self.myUtils.model_state_dict_saver(self.model,self.num_epochs,"SOC_481")

    def Tester(self,epoch=1):
        with torch.no_grad():
            total_test_error = 0
            sample_length = len(self.test_loader)
            for i, (inputs, soc_gt) in enumerate(self.test_loader):
                # if i>0:
                #     break
                inputs = inputs.float().to(self.device)
                soc_gt = soc_gt.float().to(self.device)
                outputs = self.model(inputs)
                predicted,_ = torch.max(outputs.data, 1)
                merr = torch.mean(predicted-soc_gt)
                total_test_error = abs(merr)+total_test_error
                # print(merr,total_test_error, sample_length)
            self.myUtils.writer.add_scalar("Test Error",(total_test_error*100/sample_length),epoch)
        print('Test error of the network for {} epochs is: {}'.format(epoch,total_test_error/sample_length))


if __name__ == "__main__":

    Model = ModelClass()
    Model.Trainer()
    Model.Tester(Model.num_epochs)
    