import torch.nn
from matplotlib import pyplot as plt

from torch.nn import functional as F
from torch import nn
import math


def norm(A):
    return torch.sum(A * A)


def forgetting_hook(module, grad_input, grad_output):
    if (isinstance(module, ConvGradChanger) or isinstance(module, LinearGradChanger)) and module.training is True:
        eps = 1e-15

        GA, Gi, Gw = grad_input  # gradient of A, gradient of input, gradient of weight (A=input*weight)
        """
        we want Gw become 0, since true gradient should follow from original path
        """
        if norm(Gw) > eps:
            # hook will be called multiple time, however,
            # we only want to applied it once.
            # we could detect if its second time by check Gw norm,
            # since we make is zero after first time
            Go = grad_output[0]
            if isinstance(module, ConvGradChanger):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1, 2, 3])).view(1, -1, 1, 1)
            if isinstance(module, LinearGradChanger):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1])).view(1, -1)
                """
                f=how much each neuron update its weights
                by update assumption this value for all neurons should be equal,
                to make it a multi-agent neural network.
                using this information we should scale gradient to make it equality between neurons.
                """

            f[torch.abs(f) < eps] = torch.mean(f)
            module.plot_f(f)
            # if gradients are zero(f=0), then no matter how much we scale
            # this neuron is still not going to change its input
            # results in a numerical instability
            Gr = Go / (f + eps)
            f_normalize = f / (
                   math.sqrt(torch.sum(GA * GA) / (torch.sum(Gr * Gr) + eps)) + eps)
            Grn = Go / (f_normalize + eps)
            # Gr and Grn have same direction, so they differ only by a scaler,
            # we want to keep norm of gradients similar to error-backpropagation
            return (Grn, torch.zeros_like(Gi), torch.zeros_like(Gw))  # GA->Grn, Gi->0, Gw->0



def hook(module, grad_input, grad_output):
    if (isinstance(module, ConvGradChanger) or isinstance(module, LinearGradChanger)) and module.training is True:
        eps = 1e-15

        GA, Gi, Gw = grad_input  # gradient of A, gradient of input, gradient of weight (A=input*weight)
        """
        we want Gw become 0, since true gradient should follow from original path
        """
        if norm(Gw) > eps:
            # hook will be called multiple time, however,
            # we only want to applied it once.
            # we could detect if its second time by check Gw norm,
            # since we make is zero after first time
            Go = grad_output[0]
            if isinstance(module, ConvGradChanger):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1, 2, 3])).view(1, -1, 1, 1)
            if isinstance(module, LinearGradChanger):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1])).view(1, -1)
                """
                f=how much each neuron update its weights
                by update assumption this value for all neurons should be equal,
                to make it a multi-agent neural network.
                using this information we should scale gradient to make it equality between neurons.
                """

            f[torch.abs(f) < eps] = torch.mean(f)
            module.plot_f(f)
            # if gradients are zero(f=0), then no matter how much we scale
            # this neuron is still not going to change its input
            # results in a numerical instability
            Gr = Go / (f + eps)
            Grn = Gr/4
            # Since GA have the half of the output gradient due to averaging
            return (Grn, torch.zeros_like(Gi), torch.zeros_like(Gw))  # GA->Grn, Gi->0, Gw->0


class ConvGradChanger(nn.Module):
    def __init__(self, stride, padding):
        super(ConvGradChanger, self).__init__()
        self.stride = stride
        self.padding = padding
        self.name = None

    def forward(self, A, input, weight):
        B = F.conv2d(input, weight, torch.zeros(weight.shape[0], device=weight.device),
                     stride=self.stride, padding=self.padding)
        return (A + B) / 2


class LinearGradChanger(nn.Module):
    def __init__(self):
        """ we use this module to
         add a backward hook on top of it,
         to be able to change gradients in backward pass
        """
        super(LinearGradChanger, self).__init__()
        self.name = None

    def forward(self, A, input, weight):
        B = F.linear(input, weight, torch.zeros(weight.shape[0], device=weight.device))  # in value B=A so (A+B)/2=A=B
        return (A + B) / 2


class GreedyConv2dPlain(nn.Module):
    def __init__(self, input_feature, output_feature, kernel_size, stride, padding, bias):
        super().__init__()
        self.has_bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv = nn.Conv2d(input_feature, output_feature, kernel_size, stride, bias=bias)
        self.weight = torch.nn.parameter.Parameter(data=conv.weight.clone().cuda(), requires_grad=True)
        self.register_parameter("Gconv2dweight", self.weight)
        if self.has_bias:
            self.bias = torch.nn.parameter.Parameter(data=conv.bias.clone().cuda(), requires_grad=True)
            self.register_parameter("Gconv2dbias", self.bias)
        with torch.no_grad():
            if self.has_bias:
                self.bias_initial_norm = torch.linalg.norm(self.bias.data)
            self.weigth_initial_norm = torch.linalg.matrix_norm(self.weight.data)
        self.convGradChanger = ConvGradChanger(self.stride, self.padding)

    def forward(self, input):
        A = F.conv2d(input, self.weight, torch.zeros(self.weight.shape[0], device=self.weight.device),
                     self.stride, self.padding)
        O = self.convGradChanger(A, input, self.weight)
        return O + self.bias.view(1, -1, 1, 1) if self.has_bias else O


class GreedyLinearPlain(nn.Module):
    def __init__(self, input_feature, output_feature):
        """
        A linear layer follows the theories in the paper. (each neuron act as rational
        agent trying to maximize its utility).

        :param input_feature: number of input features
        :param output_feature: number of output features
        """
        super().__init__()

        linear = nn.Linear(input_feature, output_feature)
        self.weight = torch.nn.parameter.Parameter(data=linear.weight.clone().cuda(), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(data=linear.bias.clone().cuda(), requires_grad=True)
        self.linearGradChanger = LinearGradChanger()

    def forward(self, input):
        A = F.linear(input, self.weight, torch.zeros(self.weight.shape[0], device=self.weight.device), )
        O = self.linearGradChanger(A, input, self.weight)  # in value O is equal to A, however its grads are different
        return O + self.bias.view(1, -1)


class GLinear(nn.Module):
    def __init__(self, input_size, output_size, mode="greedy", activation=None, ):
        """
        A linear Module
        :param input_size: input features
        :param output_size:  output features
        :param mode: normal, greedy
        :param activation: an optional activation applied after linear function
        """
        super().__init__()
        self.mode = mode
        self.activation = activation
        self.output_size = output_size
        if self.mode == "normal":
            self.linear = nn.Linear(input_size, output_size)
        if self.mode == "greedy":
            self.linear = GreedyLinearPlain(input_size, output_size)

    def forward(self, input: torch.tensor):
        x = self.linear(input)
        if self.activation:
            x = self.activation(x)
        return x


class GConv2d(nn.Module):
    def __init__(self, input_feature, output_feature, kernel_size, stride=1, padding=0, mode="greedy", bias=True,
                 activation=None):
        super().__init__()
        assert mode in ["greedy", "normal"]
        self.mode = mode
        self.activation = activation
        if self.mode == "normal":
            self.conv2d = torch.nn.Conv2d(input_feature, output_feature, kernel_size, stride, bias=bias)
        if self.mode == "greedy":
            self.conv2d = GreedyConv2dPlain(input_feature, output_feature, kernel_size,
                                            stride=stride, padding=padding, bias=bias)

    def forward(self, input: torch.tensor):
        x = self.conv2d(input)
        if self.activation:
            x = self.activation(x)
        return x


# model
class ClassifierMLP(torch.nn.Module):
    """ Simple MLP model
    input_feature: number of input feature
    hidden_layer_size: a list of units in hidden layer
    class_num: number of classes
    mode: normal, greedy,
    """

    def __init__(self, input_feature, hidden_layer_size, class_num, mode):
        super(ClassifierMLP, self).__init__()
        self.layers = []
        input_size = input_feature
        self.mode = mode
        for i, h in enumerate(hidden_layer_size):
            self.layers.append(GLinear(input_size, h, mode, nn.ReLU()))
            input_size = h
        self.layers.append(GLinear(input_size, class_num, mode, None))
        self.deep = nn.Sequential(*self.layers)

    def forward(self, x, t=1):
        x = torch.flatten(x, start_dim=1)
        x = self.deep(x)
        return x * t

class LeNet(torch.nn.Module):
    """
    LeNet model
    """

    def __init__(self, num_classes, mode, input_channels):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            GConv2d(input_channels, 6, kernel_size=5, stride=1, padding=0, mode=mode, activation=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            GConv2d(6, 16, kernel_size=5, stride=1, padding=0, mode=mode, activation=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2))
        features = 256 if input_channels == 1 else 400
        self.fc = GLinear(features, 120, mode=mode, activation=nn.ReLU())
        self.fc1 = GLinear(120, 84, mode=mode, activation=nn.ReLU())
        self.fc2 = GLinear(84, num_classes, mode=mode, activation=None)

    def forward(self, x, t=1):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out * t


# change model state
def set_to_eval(model):
    model.eval()


def set_to_train(model):
    model.train()


def set_name(model, prefix=""):
    for name, module in model.named_children():
        if isinstance(module, LinearGradChanger) or isinstance(module, ConvGradChanger):
            module.name = prefix
        else:
            set_name(module, prefix=prefix + "." + name)
