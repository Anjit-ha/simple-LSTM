import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()
        self.hidden_size=hidden_size
        #forgot gate
        self.fi=nn.Linear(input_size,hidden_size)
        self.fh=nn.Linear(hidden_size,hidden_size)
        #input gate
        self.gi=nn.Linear(input_size,hidden_size)
        self.gh=nn.Linear(hidden_size,hidden_size)
        #output gate
        self.qi=nn.Linear(input_size,hidden_size)
        self.qh=nn.Linear(hidden_size,hidden_size)
        #state level
        self.si=nn.Linear(input_size,hidden_size)
        self.sh=nn.Linear(hidden_size,hidden_size)
       #output
        self.ot=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)
        
    def forward(self,input_l,hidden,cell_state):
        #f_gate
        forgot=F.sigmoid(self.fi(input_l) + self.fh(hidden))
        #input_gate
        input_g=F.sigmoid(self.gi(input_l) + self.gh(hidden))
        #output_gate
        output_g=F.sigmoid(self.qi(input_l) + self.qh(hidden))
        
        state=F.tanh(self.si(input_l) + self.sh(hidden))
        cell_state=forgot*cell_state+input_g*state 
        hidden=F.tanh(cell_state)*output_g
        output=self.ot(hidden)
        output=self.softmax(output)
        return output,hidden,cell_state
    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)
    def init_cell_state(self):
        return torch.zeros(1,self.hidden_size)
