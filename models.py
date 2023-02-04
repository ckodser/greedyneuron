import torch.nn
from matplotlib import pyplot as plt

from torch.nn import functional as F
from torch import nn
import math


def norm(A):
    return torch.sum(A * A)


def hook(module, grad_input, grad_output):
    if (isinstance(module, ConvGradChanger) or isinstance(module, LinearGradChanger) or isinstance(module,
                                                                                                   LinearGradChangerExtraverts)) and module.training is True:
        eps = 1e-15
        if isinstance(module, LinearGradChangerExtraverts):
            GA, Gi, Gw, Gextra = grad_input
        else:
            GA, Gi, Gw = grad_input
        if norm(Gw) > eps:
            Go = grad_output[0]
            if isinstance(module, ConvGradChanger):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1, 2, 3])).view(1, -1, 1, 1)
            if isinstance(module, LinearGradChanger) or isinstance(module, LinearGradChangerExtraverts):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1])).view(1, -1)
            # print(module, "Number of zeros:",torch.sum(torch.abs(f)<eps).item())
            # f = f * module.born_inequality
            f[torch.abs(f) < eps] = torch.mean(f)
            assert torch.sum(torch.abs(f) < eps).item() == 0
            Gr = Go / (f + eps)
            f_normalize = f / (math.sqrt(torch.sum(GA * GA) / (torch.sum(Gr * Gr) + eps)) + eps)
            Grn = Go / (f_normalize + eps)

            # GAN = torch.sum(GA * GA).item()
            # GrnN = torch.sum(Grn * Grn).item()
            # diff = GAN - GrnN
            # # print(diff, GAN, "=>", GrnN, norm(Gw).item(), Gw.shape)
            # try:
            #     assert abs(diff) < abs(GAN / 1000)
            # except:
            #     print(module, diff, GAN, GrnN)
            #     assert abs(diff) < abs(GAN / 1000)
            if isinstance(module, LinearGradChangerExtraverts):
                return (Grn, torch.zeros_like(Gi), torch.zeros_like(Gw), torch.zeros_like(Gextra))
            else:
                return (Grn, torch.zeros_like(Gi), torch.zeros_like(Gw))


class ConvGradChanger(nn.Module):
    def __init__(self, stride, padding):
        super(ConvGradChanger, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, A, input, weight):
        B = F.conv2d(input, weight, torch.zeros(weight.shape[0], device=weight.device),
                     stride=self.stride, padding=self.padding)
        return (A + B) / 2


class LinearGradChanger(nn.Module):
    def __init__(self):
        super(LinearGradChanger, self).__init__()

    def forward(self, A, input, weight):
        B = F.linear(input, weight, torch.zeros(weight.shape[0], device=weight.device))
        return (A + B) / 2


def extravertsLinear(input, weight, extravertish):
    fake_weight_non_detached = torch.pow(torch.abs(weight), extravertish) * torch.sign(weight)
    fake_weight = fake_weight_non_detached.detach() + weight - weight.detach()

    fake_output_non_detached = F.linear(input, fake_weight, torch.zeros(weight.shape[0], device=weight.device))
    real_output = F.linear(input, weight, torch.zeros(weight.shape[0], device=weight.device))
    fake_output = fake_output_non_detached - fake_output_non_detached.detach() + real_output.detach()
    return fake_output


class LinearGradChangerExtraverts(nn.Module):
    def __init__(self, neuron_num):
        super(LinearGradChangerExtraverts, self).__init__()
        # self.born_inequality = torch.clip(torch.exp(torch.randn(neuron_num)), 0.7,1.5).cuda()

    def forward(self, A, input, weight, extravertish):
        B = extravertsLinear(input, weight, extravertish)
        return (A + B) / 2


class GreedyConv2dPlain(nn.Module):
    def __init__(self, input_feature, output_feature, kernel_size, stride, padding):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv = nn.Conv2d(input_feature, output_feature, kernel_size, stride)
        self.weight = torch.nn.parameter.Parameter(data=conv.weight.clone(), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(data=conv.bias.clone(), requires_grad=True)
        self.register_parameter("Gconv2dweight", self.weight)
        self.register_parameter("Gconv2dbias", self.bias)
        with torch.no_grad():
            self.bias_initial_norm = torch.linalg.norm(self.bias.data)
            self.weigth_initial_norm = torch.linalg.matrix_norm(self.weight.data)
        self.convGradChanger = ConvGradChanger(self.stride, self.padding)

    def forward(self, input):
        A = F.conv2d(input, self.weight, torch.zeros(self.weight.shape[0], device=self.weight.device),
                     self.stride, self.padding)
        O = self.convGradChanger(A, input, self.weight)
        return O + self.bias.view(1, -1, 1, 1)

class GreedyConv2dPlainExtraverts(nn.Module):
    def __init__(self, input_feature, output_feature, kernel_size, extravert_mult, extravert_bias,
                 stride, padding):
        super().__init__()
        raise NotImplementedError
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        conv = nn.Conv2d(input_feature, output_feature, kernel_size, stride)
        self.weight = torch.nn.parameter.Parameter(data=conv.weight.clone(), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(data=conv.bias.clone(), requires_grad=True)
        self.register_parameter("Gconv2dweight", self.weight)
        self.register_parameter("Gconv2dbias", self.bias)
        with torch.no_grad():
            self.bias_initial_norm = torch.linalg.norm(self.bias.data)
            self.weigth_initial_norm = torch.linalg.matrix_norm(self.weight.data)
        self.convGradChanger = ConvGradChanger(self.stride, self.padding)

    def forward(self, input):
        A = F.conv2d(input, self.weight, torch.zeros(self.weight.shape[0], device=self.weight.device),
                     self.stride, self.padding)
        O = self.convGradChanger(A, input, self.weight)
        return O + self.bias.view(1, -1, 1, 1)

class GreedyLinearPlain(nn.Module):
    def __init__(self, input_feature, output_feature):
        super().__init__()

        linear = nn.Linear(input_feature, output_feature)
        self.weight = torch.nn.parameter.Parameter(data=linear.weight.clone(), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(data=linear.bias.clone(), requires_grad=True)
        with torch.no_grad():
            self.bias_initial_norm = torch.linalg.norm(self.bias.data)
            self.weigth_initial_norm = torch.linalg.matrix_norm(self.weight.data)
        self.linearGradChanger = LinearGradChanger()

    def forward(self, input):
        A = F.linear(input, self.weight, torch.zeros(self.weight.shape[0], device=self.weight.device), )
        O = self.linearGradChanger(A, input, self.weight)
        return O + self.bias.view(1, -1)


class GreedyLinearPlainExtraverts(GreedyLinearPlain):
    def __init__(self, input_feature, output_feature, extravert_mult, extravert_bias):
        super(GreedyLinearPlainExtraverts, self).__init__(input_feature, output_feature)
        self.linearGradChangerExtraverts = LinearGradChangerExtraverts(output_feature)
        self.extravertish = torch.nn.parameter.Parameter(
            data=torch.exp(torch.randn(output_feature, 1)*extravert_mult)+extravert_bias,
            requires_grad=False)
        plt.hist(self.extravertish.numpy())
        plt.show()

    def forward(self, input):
        A = extravertsLinear(input, self.weight, self.extravertish)
        O = self.linearGradChangerExtraverts(A, input, self.weight, self.extravertish)
        return O + self.bias.view(1, -1)


class GLinear(nn.Module):
    def __init__(self, input_size, output_size, mode, activation, extravert_mult=1, extravert_bias=0):
        super().__init__()
        self.mode = mode
        self.activation = activation
        self.output_size = output_size
        if self.mode == "normal":
            self.linear = nn.Linear(input_size, output_size)
        if self.mode == "greedy":
            self.linear = GreedyLinearPlain(input_size, output_size)
        if self.mode == "greedyExtraverts":
            self.linear = GreedyLinearPlainExtraverts(input_size, output_size, extravert_mult, extravert_bias)

        if self.mode == "intel":
            self.bn = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, input: torch.tensor):
        x = self.linear(input)
        if self.activation:
            x = self.activation(x)
        if self.mode == "intel":
            x = self.bn(x)
        return x


class GConv2d(nn.Module):
    def __init__(self, input_feature, output_feature, kernel_size, stride, padding, mode, activation, extravert_mult, extravert_bias):
        super().__init__()
        assert mode in ["greedy", "normal", "intel"]
        self.mode = mode
        self.activation = activation
        if self.mode == "normal":
            self.conv2d = torch.nn.Conv2d(input_feature, output_feature, kernel_size, stride)
        if self.mode == "greedy":
            self.conv2d = GreedyConv2dPlain(input_feature, output_feature, kernel_size,
                                            stride=stride, padding=padding)
        if self.mode == "greedyExtraverts":
            self.linear = GreedyConv2dPlainExtraverts(input_feature, output_feature, kernel_size, extravert_mult,
                                                      extravert_bias, stride=stride, padding=padding)

        if self.mode == "intel":
            self.bn = nn.BatchNorm2d(output_feature)

    def forward(self, input: torch.tensor):
        x = self.conv2d(input)
        if self.activation:
            x = self.activation(x)
        if self.mode == "intel":
            x = self.bn(x)
        return x


# model
class ClassifierMLP(torch.nn.Module):
    def __init__(self, hidden_layer_size, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierMLP, self).__init__()
        self.layers = []
        input_size = 784
        self.mode = mode
        for i, h in enumerate(hidden_layer_size):
            self.layers.append(GLinear(input_size, h, mode, nn.ReLU(), extravert_mult, extravert_bias))
            input_size = h
        self.layers.append(GLinear(input_size, class_num, mode, None, extravert_mult, extravert_bias))
        self.deep = nn.Sequential(*self.layers)

    def forward(self, x, t=1):
        x = torch.flatten(x, start_dim=1)
        x = self.deep(x)
        return x * t


class ClassifierCNN(torch.nn.Module):
    def __init__(self, hidden_featuer_num, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierCNN, self).__init__()
        self.layers = []
        input_feature = 1
        self.mode = mode
        for i, h in enumerate(hidden_featuer_num):
            self.layers.append(GConv2d(input_feature, h, 3, 1, 0, mode, nn.ReLU(), extravert_mult, extravert_bias))
            input_feature = h
        self.layers.append(torch.nn.AvgPool2d(28 - 2 * len(hidden_featuer_num)))
        self.layers.append(torch.nn.Flatten(start_dim=1))
        self.layers.append(GLinear(hidden_featuer_num[-1], class_num, mode, None, extravert_mult, extravert_bias))
        self.deep = nn.Sequential(*self.layers)

    def forward(self, x, t=1):
        x = self.deep(x)
        return x * t

class ClassifierCNNWide(torch.nn.Module):
    def __init__(self, hidden_featuer_num, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierCNNWide, self).__init__()
        self.layers = []
        input_feature = 1
        self.mode = mode
        for i, h in enumerate(hidden_featuer_num):
            self.layers.append(GConv2d(input_feature, h, 7, 1, 0, mode, nn.Sequential(
                nn.ReLU(), nn.MaxPool2d(2)), extravert_mult, extravert_bias))
            input_feature = h
        self.layers.append(torch.nn.AvgPool2d(2))
        self.layers.append(torch.nn.Flatten(start_dim=1))
        self.layers.append(GLinear(hidden_featuer_num[-1], class_num, mode, None, extravert_mult, extravert_bias))
        self.deep = nn.Sequential(*self.layers)


    def forward(self, x, t=1):
        x = self.deep(x)
        return x * t

def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    out = self.relu(out)
    out = self.fc1(out)
    out = self.relu1(out)
    out = self.fc2(out)
    return out


class LeNet(torch.nn.Module):
    def __init__(self, num_classes, mode):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            GConv2d(1, 6, kernel_size=5, stride=1, padding=0, mode=mode, activation=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            GConv2d(6, 16, kernel_size=5, stride=1, padding=0, mode=mode, activation=nn.ReLU()),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = GLinear(256, 120, mode=mode, activation=nn.ReLU())
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
