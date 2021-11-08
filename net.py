import torch
from torch import nn as nn
import torch.optim as optim
import os
import joblib


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, lr=0.005, use_gpu=False):
        self.use_gpu = use_gpu
        cuda = use_gpu
        self.device = torch.device('cuda' if cuda else 'cpu')
        if not cuda:
            torch.set_num_threads(10)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_norms = []
        self.conv_layers = []
        self.common_layers = []
        self.policy_layers = []
        self.value_layers = []
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=-1)
        self.compile_fully()
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def compile(self, common_sizes, policy_sizes, value_sizes, in_size=None):
        if in_size is None:
            in_size = self.input_dim[0]
        for i, next_size in enumerate(common_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("_fc{}".format(i), fc)
            self.common_layers.append(fc)
        in_size_value = in_size
        for i, next_size in enumerate(policy_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("_fc_policy{}".format(i), fc)
            self.policy_layers.append(fc)
        in_size = in_size_value
        for i, next_size in enumerate(value_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("_fc_value{}".format(i), fc)
            self.value_layers.append(fc)

        out_1 = self.policy = nn.Linear(policy_sizes[-1], self.output_dim)
        self.__setattr__("_policy{}", out_1)
        out_2 = self.value = nn.Linear(value_sizes[-1], 1)
        self.__setattr__("_value{}", out_2)
        return out_1, out_2

    def compile_fully(self):
        sizes = [20]
        value_sizes = [4]
        policy_sizes = [10]
        self.activation = self.selu
        return self.compile(common_sizes=sizes, policy_sizes=policy_sizes, value_sizes=value_sizes)

    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        P, v = self.forward(x)
        return get_numpy(P), get_numpy(v)
    
    def predict_one(self, x):
        P, v = self.predict(x[None])
        return P[0], v[0][0]
        
    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self,):
        return self.state_dict()

    def save(self, path):
        state = self.state_dict()
        dirname = os.path.dirname(path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(state, path)

    def load(self, load_path):
        state_dict = joblib.load(os.path.expanduser(load_path))
        self.load_state_dict(state_dict)

    def load_weights(self, load_path):
        self.load(load_path)

    def forward(
            self,
            obs
    ):
        """
        :param obs: Observation
        """

        h = obs.to(self.device)
        for i, conv in enumerate(self.conv_layers):
            h = self.relu(self.batch_norms[i](conv(h)))
        if len(self.conv_layers) > 0:
            h = h.view(h.size(0), -1)  ##  flatten

        for i, fc in enumerate(self.common_layers):
            h = self.activation(fc(h))
        h_v = h
        for i, fc in enumerate(self.policy_layers):
            h = self.activation(fc(h))
        h_p = h
        h = h_v
        for i, fc in enumerate(self.value_layers):
            h = self.activation(fc(h))
        h_v = h

        policy = self.softmax(self.policy(h_p))
        value = self.value(h_v)
        return policy, value
