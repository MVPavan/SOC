import torch
import torch.nn as nn

class SOCNN():
    def __init__(self,):
        n_input, n_hidden, n_output = 4, 4, 1
        self.x = torch.randn((1, n_input))
        self.y = torch.randn((1, n_output))
        self.w1 = torch.randn(n_input, n_hidden)
        self.w2 = torch.randn(n_hidden, n_output)
        self.b1 = torch.randn((1, n_hidden))
        self.b2 = torch.randn((1, n_output))
    
    def sigmoid_activation(self,z):
        return 1 / (1 + torch.exp(-z))
    
    
    
    def main(self,):
        self.z1 = torch.mm(x, self.w1) + self.b1
        self.a1 = self.sigmoid_activation(self.z1)
        self.z2 = torch.mm(self.a1, self.w2) + self.b2
        self.output = self.sigmoid_activation(self.z2)
        
        self.loss = self.actual_soc - self.output
        
        