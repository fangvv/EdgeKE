# 将祖目录加入临时路径
import sys
sys.path.append("../..")
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ResNet import ResNet
from functions import my_functions
from torchsummary import summary

def get_model(num_classes=100):
    main_model = ResNet.ResNet18(num_classes=num_classes)
    branch_model = []
    # branch_model_0 = ResNet_0(BasicBlock, [2,2], num_classes=num_classes)
    # branch_model_1 = ResNet_1(BasicBlock, [2,2,2], num_classes=num_classes)
    branch_model_0 = ResNet_1(BasicBlock, [2,2,2], num_classes=num_classes)
    branch_model_1 = ResNet_2(BasicBlock, [2,2,2,1], num_classes=num_classes)
    branch_model.append(branch_model_0)
    branch_model.append(branch_model_1)
    branch_model.append(main_model)
    # 返回主干网络 与 分支网络
    return main_model, branch_model

# 返回教师网络
def get_teacher_model(num_classes=100):
    teacher_model = ResNet.ResNet34(num_classes=num_classes)
    return teacher_model



# def get_model():
#     main_model = ResNet.ResNet18()
#     branch_model = []
#     branch_model_0 = ResNet_0(BasicBlock, [2,2])
#     branch_model_1 = ResNet_1(BasicBlock, [2,2,2])
#     branch_model.append(branch_model_0)
#     branch_model.append(branch_model_1)
#     branch_model.append(main_model)
#     # 返回主干网络 与 分支网络
#     return main_model, branch_model
#
# # 返回教师网络
# def get_teacher_model():
#     teacher_model = ResNet.ResNet34()
#     return teacher_model

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 分支 0
class ResNet_0(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_0, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.linear = nn.Linear(128*block.expansion, 128)
        self.linear1 = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 16)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.linear1(out)
        return out

#分支 1
class ResNet_1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_1, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, 256)
        self.linear1 = nn.Linear(256, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        # print ("mild out:",out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.linear1(out)
        return out

class ResNet_2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_2, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 512)
        self.linear1 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.linear1(out)
        return out


if __name__ == "__main__":
    main_model, branch_model = get_model(num_classes=100)
    teacher_model = get_teacher_model(num_classes=100)

    # summary(branch_model[0], input_size=(3, 32, 32))
    # summary(branch_model[1], input_size=(3, 32, 32))
    # summary(branch_model[2], input_size=(3, 32, 32))
    summary(teacher_model, input_size=(3, 32, 32))
    # print (branch_model[0])
    # model = ResNet_2(BasicBlock, [2,2,2,1], num_classes=100)
    # summary(model, input_size=(3, 32, 32))
    # for i, data in enumerate(model.named_parameters()):
    #     name, param = data
    #     print(i, name, param.size())
