# 代码测试的代码，没必要看


# 冻结参数
from torchsummary import summary
from model.ResNet import get_ResNet_model
import math
import time
import torch
import torch.nn as nn
from functions.branch_functions import Eval_BranchyNet
from functions.my_functions import Eval_model
from functions import  my_functions
from datasets import get_cifar_100
from torchstat import stat


# def get_threshold(outputs):
#     # Method 1
#     softmax = nn.Softmax(dim=1)
#     result = softmax(outputs)
#     temp = torch.max(result.data, 1)
#     temp = temp[0].numpy()
#     return temp[0]
#
# def get_Exit_Threshold(branch_model, test_loader, bili):
#     Exit_Threshold = []
#     model_num = len(branch_model)
#     for i in range(model_num):
#         branch_model[i].eval()
#         Threshold = []
#         total = 0
#         for key, data in enumerate(test_loader):
#             inputs, labels = data
#             total += labels.size(0)
#             output = branch_model[i](inputs)
#             Threshold.append(get_threshold(output))
#         Threshold.sort(reverse=True)
#         flag = int(total * bili[i])
#         Exit_Threshold.append(Threshold[flag])
#     print (Exit_Threshold)
#     return Exit_Threshold

if __name__ == "__main__":

    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)
    # teacher_model = get_VGGNet_model.get_teacher_model(num_classes=100)
    # main_model, branch_model = get_VGGNet_model.get_model(num_classes=100)
    # for i in branch_model[0].state_dict():
    #     print (i)
    # for i, data in enumerate(branch_model[0].named_parameters()):
    #     name, param = data
    #     print (i, name, param.size())
    # summary(branch_model[1], input_size=(3, 32, 32))
    summary(teacher_model, input_size=(3, 32, 32))
    # stat(branch_model[2], (3, 32, 32))


    # print ("ResNet Test ")
    # Project_dir = my_functions.get_project_dir()
    # main_model, branch_model = get_ResNet_model.get_model()
    # teacher_model = get_ResNet_model.get_teacher_model()
    #
    # train_loader, test_loader = get_cifar_10.get_data(train_batch_size=128, test_batch_size=1)
    # model_num = len(branch_model)
    #
    # teacher_dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_1_checkpoint.tar"
    # teacher_checkpoint = torch.load(teacher_dir, map_location=lambda storage, loc: storage)
    # teacher_model.load_state_dict(teacher_checkpoint["state_dict"])
    #
    # for i in range(model_num):
    #     dir = Project_dir + "/model/ResNet/KD/ResNet_model_" + str(i) + "_checkpoint.tar"
    #     checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    #     branch_model[i].load_state_dict(checkpoint["state_dict"])
    # bili = [0.7, 0.2, 0.1]
    # # exit_threshold1 = get_Exit_Threshold(branch_model, test_loader, bili)
    # exit_threshold = [0.998, 0.99, 0]
    # voting_weight = [0.2, 0.5, 0.3]
    # Eval_BranchyNet(branch_model, exit_threshold, voting_weight, test_loader)
    # pre_acc, loss, total_time, once_time = Eval_model(teacher_model, test_loader)
    # print("Eval info, Acc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
    #       % (pre_acc, loss, total_time, once_time * 1000))

    # print("VGGNet Test ")
    # Project_dir = my_functions.get_project_dir()
    # main_model, branch_model = get_VGGNet_model.get_model()
    # teacher_model = get_VGGNet_model.get_teacher_model()
    #
    # train_loader, test_loader = get_cifar_10.get_data(train_batch_size=128, test_batch_size=1)
    # model_num = len(branch_model)
    #
    # teacher_dir = Project_dir + "/model/VGGNet/KD/VGGNet_Teacher_model_1_checkpoint.tar"
    # teacher_checkpoint = torch.load(teacher_dir, map_location=lambda storage, loc: storage)
    # teacher_model.load_state_dict(teacher_checkpoint["state_dict"])
    #
    # for i in range(model_num):
    #     dir = Project_dir + "/model/VGGNet/KD/VGGNet_model_" + str(i) + "_checkpoint.tar"
    #     checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    #     branch_model[i].load_state_dict(checkpoint["state_dict"])
    # bili = [0.8, 0.15, 0.05]
    # # exit_threshold1 = get_Exit_Threshold(branch_model, test_loader, bili)
    # exit_threshold = [0.999, 0.999, 0]
    # voting_weight = [0.2, 0.5, 0.3]
    # Eval_BranchyNet(branch_model, exit_threshold, voting_weight, test_loader)
    # pre_acc, loss, total_time, once_time = Eval_model(teacher_model, test_loader)
    # print("Eval info, Acc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
    #       % (pre_acc, loss, total_time, once_time * 1000))


