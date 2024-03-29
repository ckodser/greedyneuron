import torch.nn
from matplotlib import pyplot as plt

from torch.nn import functional as F
from torch import nn
import math
import wandb


def norm(A):
    return torch.sum(A * A)


def hook(module, grad_input, grad_output):
    if (isinstance(module, ConvGradChanger) or isinstance(module, LinearGradChanger) or
        isinstance(module, LinearGradChangerExtraverts)) and module.training is True:
        eps = 1e-15
        if isinstance(module, LinearGradChangerExtraverts):
            GA, Gi, Gw, Gextra = grad_input
        else:
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
            if isinstance(module, LinearGradChanger) or isinstance(module, LinearGradChangerExtraverts):
                f = torch.sqrt(torch.sum(Gw * Gw, dim=[1])).view(1, -1)
                """
                f=how much each neuron update its weights
                by update assumption this value for all neurons should be equal,
                to make it a multi-agent neural network.
                using this information we should scale gradient to make it equality between neurons.
                """

            f[torch.abs(f) < eps] = torch.mean(f)
            module.plot_f(f, GA)
            # if gradients are zero(f=0), then no matter how much we scale
            # this neuron is still not going to change its input
            # results in a numerical instability
            Gr = Go / (f + eps)
            # f_normalize = f / (
            #        math.sqrt(torch.sum(GA * GA) / (torch.sum(Gr * Gr) + eps)) + eps)
            # Grn = Go / (f_normalize + eps)
            Grn = Gr / 2
            # Gr and Grn are same but they differ only by a scaler,
            # we want to keep norm of gradients similar to error-backpropagation
            if isinstance(module, LinearGradChangerExtraverts):
                return (Grn, torch.zeros_like(Gi), torch.zeros_like(Gw), torch.zeros_like(Gextra))
            else:
                return (Grn, torch.zeros_like(Gi), torch.zeros_like(Gw))  # GA->Grn, Gi->0, Gw->0


class ConvGradChanger(nn.Module):
    def __init__(self, stride, padding):
        super(ConvGradChanger, self).__init__()
        self.stride = stride
        self.padding = padding
        self.tracking = False
        self.name = None

    def plot_f(self, f, raw_grad):
        if self.tracking:
            c = 1 / f
            # c tracking
            score = torch.flatten(c)
            data = [[score[i]] for i in range(score.shape[0])]
            table = wandb.Table(data=data, columns=["utility"])
            wandb.log({f"c_tracking/w_{self.name}": wandb.plot.histogram(table, "value", title="c_tracking"), })

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
        self.tracking = False
        self.name = None

    def plot_f(self, f, raw_grad):
        if self.tracking:
            c = 1 / f
            # c tracking
            score = torch.flatten(c).cpu()
            torch.save(score, f"T{self.name}_{self.current_epoch}")
            torch.save(raw_grad ,f"Grad{self.name}_{self.current_epoch}")
            # plt.hist(score, bins=20)
            # plt.savefig(f"M{self.name}_{self.current_epoch}.png")
            # print(f"save fig: M{self.name}_{self.current_epoch}.png")
            # plt.show()
            # table = wandb.Table(data=data, columns=["utility"])
            # wandb.log({f"c_tracking/w_{self.name}": wandb.plot.histogram(table, "value", title="c_tracking"), })

    def forward(self, A, input, weight):
        B = F.linear(input, weight, torch.zeros(weight.shape[0], device=weight.device))  # in value B=A so (A+B)/2=A=B
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
        return (O + self.bias.view(1, -1, 1, 1)) if self.has_bias else O


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
    def __init__(self, input_feature, output_feature, residual_initialization=False, bias: bool = True):
        """
        A linear layer follows the theories in the paper. (each neuron act as rational
        agent trying to maximize its utility).

        :param input_feature: number of input features
        :param output_feature: number of output features
        :param residual_initialization: if true, bias = zero, weight = eye at initialization.
         it simulates a residual connection
        """
        super().__init__()
        self.has_bias = bias
        linear = nn.Linear(input_feature, output_feature)
        self.weight = torch.nn.parameter.Parameter(data=linear.weight.clone().cuda(), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(data=linear.bias.clone().cuda(), requires_grad=True)
        if residual_initialization:
            if input_feature == output_feature:
                self.weight.data = torch.eye(self.weight.data.shape[0])
                self.bias.data = torch.zeros_like(self.bias.data)
            if output_feature == 10:
                self.weight.data = torch.zeros_like(self.weight.data)
                self.bias.data = torch.zeros_like(self.bias.data)

        # for tracking
        with torch.no_grad():
            self.bias_initial_norm = torch.linalg.norm(self.bias.data)
            self.weigth_initial_norm = torch.linalg.matrix_norm(self.weight.data)

        self.linearGradChanger = LinearGradChanger()

    def forward(self, input):
        A = F.linear(input, self.weight, torch.zeros(self.weight.shape[0], device=self.weight.device), )
        O = self.linearGradChanger(A, input, self.weight)  # in value O is equal to A, however its grads are different
        return (O + self.bias.view(1, -1)) if self.has_bias else O


class GreedyLinearPlainExtraverts(GreedyLinearPlain):
    def __init__(self, input_feature, output_feature, extravert_mult, extravert_bias):
        super(GreedyLinearPlainExtraverts, self).__init__(input_feature, output_feature)
        self.linearGradChangerExtraverts = LinearGradChangerExtraverts(output_feature)
        self.extravertish = torch.nn.parameter.Parameter(
            data=torch.exp(torch.randn(output_feature, 1) * extravert_mult) + extravert_bias,
            requires_grad=False)
        plt.hist(self.extravertish.numpy())
        plt.show()

    def forward(self, input):
        A = extravertsLinear(input, self.weight, self.extravertish)
        O = self.linearGradChangerExtraverts(A, input, self.weight, self.extravertish)
        return O + self.bias.view(1, -1)


class GLinear(nn.Module):
    def __init__(self, input_size, output_size, mode="greedy", activation=None, extravert_mult=1, extravert_bias=0,
                 bias: bool = True):
        """
        A linear Module
        :param input_size: input features
        :param output_size:  output features
        :param mode: normal, greedy, greedyExtraverts, intel
        :param activation: an optional activation applied after linear function
        :param extravert_mult: used for greedyExtraverts
        :param extravert_bias: used for greedyExtraverts
        """
        super().__init__()
        self.mode = mode
        self.activation = activation
        self.output_size = output_size
        if self.mode == "normal":
            self.linear = nn.Linear(input_size, output_size, bias=bias)
        if self.mode == "greedy":
            self.linear = GreedyLinearPlain(input_size, output_size, bias=bias)
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
    def __init__(self, input_feature, output_feature, kernel_size, stride=1, padding=0, mode="greedy",
                 bias: bool = True,
                 activation=None, extravert_mult=1, extravert_bias=0):
        super().__init__()
        assert mode in ["greedy", "normal", "intel"]
        self.mode = mode
        self.activation = activation
        if self.mode == "normal":
            self.conv2d = torch.nn.Conv2d(input_feature, output_feature, kernel_size,
                                          stride=stride, padding=padding, bias=bias)
        if self.mode == "greedy":
            self.conv2d = GreedyConv2dPlain(input_feature, output_feature, kernel_size,
                                            stride=stride, padding=padding, bias=bias)
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
    """ Simple MLP model
    input_feature: number of input feature
    hidden_layer_size: a list of units in hidden layer
    class_num: number of classes
    mode: normal, greedy, greedyExtraverts, intel
    extravert_mult, extravert_bias: used when mode is greedyExtraverts
    """

    def __init__(self, input_feature, hidden_layer_size, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierMLP, self).__init__()
        self.layers = []
        input_size = input_feature
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
    def __init__(self, input_feature, hidden_featuer_num, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierCNN, self).__init__()
        self.layers = []
        input_feature = input_feature
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


class ClassifierCNNDeep(torch.nn.Module):
    def __init__(self, input_feature, hidden_featuer_num, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierCNNDeep, self).__init__()
        self.layers = []
        input_feature = input_feature
        self.mode = mode
        for i, h in enumerate(hidden_featuer_num):
            self.layers.append(GConv2d(input_feature, h, 5, 1, 0, mode, nn.ReLU(), extravert_mult, extravert_bias))
            input_feature = h
        self.layers.append(torch.nn.AvgPool2d(4))
        self.layers.append(torch.nn.Flatten(start_dim=1))
        self.layers.append(GLinear(hidden_featuer_num[-1], class_num, mode, None, extravert_mult, extravert_bias))
        self.deep = nn.Sequential(*self.layers)

    def forward(self, x, t=1):
        x = self.deep(x)
        return x * t


class ClassifierCNNWide(torch.nn.Module):
    def __init__(self, input_feature, hidden_featuer_num, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierCNNWide, self).__init__()
        self.layers = []
        input_feature = input_feature
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


class ClassifierCNNShit(torch.nn.Module):
    def __init__(self, input_feature, hidden_featuer_num, class_num, mode, extravert_mult, extravert_bias):
        super(ClassifierCNNShit, self).__init__()
        self.layers = []
        input_feature = input_feature
        self.mode = mode
        for i, h in enumerate(hidden_featuer_num):
            self.layers.append(GConv2d(input_feature, h, 11, 1, 0, mode, nn.Sequential(
                nn.ReLU()), extravert_mult, extravert_bias))
            input_feature = h
        self.layers.append(torch.nn.AvgPool2d(2))
        self.layers.append(torch.nn.Flatten(start_dim=1))
        self.layers.append(GLinear(hidden_featuer_num[-1], class_num, mode, None, extravert_mult, extravert_bias))
        self.deep = nn.Sequential(*self.layers)

    def forward(self, x, t=1):
        x = self.deep(x)
        return x * t


# def forward(self, x):
#     out = self.layer1(x)
#     out = self.layer2(out)
#     out = out.reshape(out.size(0), -1)
#     out = self.fc(out)
#     out = self.relu(out)
#     out = self.fc1(out)
#     out = self.relu1(out)
#     out = self.fc2(out)
#     return out


class LeNet(torch.nn.Module):
    """
    LeNet model
    """

    def __init__(self, num_classes, mode, input_channels, extravert_mult, extravert_bias):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            GConv2d(input_channels, 6, kernel_size=5, stride=1, padding=0, mode=mode, activation=nn.ReLU(),
                    extravert_mult=extravert_mult, extravert_bias=extravert_bias),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            GConv2d(6, 16, kernel_size=5, stride=1, padding=0, mode=mode, activation=nn.ReLU(),
                    extravert_mult=extravert_mult, extravert_bias=extravert_bias),
            nn.MaxPool2d(kernel_size=2, stride=2))
        features = 256 if input_channels == 1 else 400
        self.fc = GLinear(features, 120, mode=mode, activation=nn.ReLU(),
                          extravert_mult=extravert_mult, extravert_bias=extravert_bias)
        self.fc1 = GLinear(120, 84, mode=mode, activation=nn.ReLU(),
                           extravert_mult=extravert_mult, extravert_bias=extravert_bias)
        self.fc2 = GLinear(84, num_classes, mode=mode, activation=None,
                           extravert_mult=extravert_mult, extravert_bias=extravert_bias)

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


def set_c_tracking(model, value, epoch=0):
    for name, module in model.named_children():
        if isinstance(module, LinearGradChanger) or isinstance(module, ConvGradChanger):
            module.tracking = value
            module.current_epoch = epoch
            print(name, " -> ", value)
        else:
            set_c_tracking(module, value, epoch)


def set_name(model, prefix=""):
    for name, module in model.named_children():
        if isinstance(module, LinearGradChanger) or isinstance(module, ConvGradChanger):
            module.name = prefix
        else:
            set_name(module, prefix=prefix + "." + name)
