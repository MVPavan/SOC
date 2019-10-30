import os
import torch
from torch.utils.tensorboard import SummaryWriter

class MyUtils():
    def __init__(self,save_path_dir="./outputs"):
        self.save_path_dir=save_path_dir
        self.writer = SummaryWriter(os.path.join(save_path_dir,"Logs"))

    def checkpoint_saver(self,model,optimizer,epoch,batch,loss,save_name=""):
        if len(save_name)==0:
            save_name = "checkpoint_{}_{}_{}.ckpt".format(epoch,batch,loss)
        else:
            save_name = "{}_checkpoint_{}_{}_{}.ckpt".format(save_name,epoch,batch,loss)
        save_path = os.path.join(self.save_path_dir,save_name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, save_path)

    def model_saver(self,model,epoch,batch,save_name=""):
        if len(save_name)==0:
            save_name = "model_{}.pth".format(epoch)
        else:
            save_name = "{}_model_{}.pth".format(save_name,epoch)
        save_path = os.path.join(self.save_path_dir,save_name)

        torch.save(model,save_path)

    def model_state_dict_saver(self,model,epoch,save_name=""):
        if len(save_name)==0:
            save_name = "model_state_dict_{}.pth".format(epoch)
        else:
            save_name = "{}_model_state_dict_{}.pth".format(save_name,epoch)
        save_path = os.path.join(self.save_path_dir,save_name)

        torch.save(model.state_dict,save_path)