# 代码测试的代码，没必要看

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import time, threading, sys, socket, json
from functions import my_functions
from model.ResNet import get_ResNet_model, get_inference_model
from functions.my_functions import Train_model, Eval_model
from torchsummary import summary
from functions.my_functions import Knowledge_distillation
from functions.branch_functions import Train_BranchyNet_Asynchronous, Train_BranchyNet_Synchronization, Eval_BranchyNet, \
    get_Exit_Threshold, Train_BranchyNet_Asynchronous_KD, Train_BranchyNet_Asynchronous_KD_Back, Train_BranchyNet_Asynchronous_Back, \
    get_threshold, get_voting_softmax
from datasets import get_cifar_100

# 复制网络层参数
def copy_layer_param(old_model, new_model, layer_count):

    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    # 遍历赋值
    for i ,data in enumerate(old_model.named_parameters()):
    # for i ,data in enumerate(new_model.named_parameters()):
        name, param = data
        if (i >= layer_count[0] and i <= layer_count[1]):
            new_state_dict[name] = old_state_dict[name]
    new_model.load_state_dict(new_state_dict)

# def Eval_Branch_model(exit_threshold = [0.95,0.97,0.98], voting_weight = [0.30, 0.30, 0.40]):
# Branch_model性能测试
def Eval_Branch_model():

    exit_threshold = [0.96, 0.00, 0.98]
    voting_weight = [0.30, 0.30, 0.40]
    test_num = 9990
    model_num = 3
    softmax = nn.Softmax(dim=1)
    Project_dir = my_functions.get_project_dir()
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    model_0, model_1, model_2, model_3, model_4 = get_inference_model.get_model(num_classes=100)

    for i in range(model_num):
        # print (i)
        dir = Project_dir + "/model/ResNet/KD/ResNet_As_model_" + str(i) + "_checkpoint.tar"
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
        branch_model[i].load_state_dict(checkpoint["state_dict"])
        if i == 0:
            model_0.load_state_dict(checkpoint["state_dict"])
            model_1.load_state_dict(checkpoint["state_dict"])
        elif i == 1:
            model_2.load_state_dict(checkpoint["state_dict"])
            model_3.load_state_dict(checkpoint["state_dict"])
        else:
            model_4.load_state_dict(checkpoint["state_dict"])

    # 初始化
    # copy_layer_param(branch_model[0], model_0, [0, 29])
    # copy_layer_param(branch_model[0], model_1, [30, 48])
    # copy_layer_param(branch_model[1], model_2, [30, 44])
    # copy_layer_param(branch_model[1], model_3, [45, 57])
    # copy_layer_param(branch_model[2], model_4, [45, 61])

    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    # 将model设置为eval模式，防止模型参数变化
    for i in range(model_num):
        branch_model[i].eval()

    if torch.cuda.is_available():
        model_0 = model_0.cuda()
        model_1 = model_1.cuda()
        model_2 = model_2.cuda()
        model_3 = model_3.cuda()
        model_4 = model_4.cuda()
        branch_model[0] = branch_model[0].cuda()
        branch_model[1] = branch_model[1].cuda()
        branch_model[2] = branch_model[2].cuda()

    model_list = []
    model_list.append(model_0)
    model_list.append(model_1)
    model_list.append(model_2)
    model_list.append(model_3)
    model_list.append(model_4)

    train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=1)

    # 测试完整网络
    # for i in range(model_num):
    #     print ("Model: ", i)
    #     start_time = time.time()
    #     for num, data in enumerate(test_loader, 0):
    #         if num < test_num:
    #             inputs, labels = data
    #             inputs, labels = Variable(inputs), Variable(labels)
    #             out = branch_model[i](inputs)
    #         else:
    #             break
    #     end_time = time.time()
    #     print("time : %2.2d s" % (end_time - start_time))

    # 分段测试
    # 中间运行记录
    total = 0
    run_time = []
    exit_count = []
    pre_right_count = []
    voting_count = 0
    voting_time = 0.0
    voting_right_count = 0
    # 初始化
    for i in range(model_num):
        run_time.append(0.0)
        exit_count.append(0)
        pre_right_count.append(0)

    temp_start_time = time.time()
    # 开始测试BranchyNet
    for key, data in enumerate(test_loader):
        # if key == test_num:
        #     break
        # 阈值和中间输出值
        Threshold = []
        output_list = []
        output_temp = []
        for i in range(model_num):
            Threshold.append(0.0)
            output_list.append(0.0)
            output_temp.append(0.0)

        # start_time = time.time()
        inputs, labels = data
        # 使用cuda，并且有cuda资源
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # a = softmax(branch_model[0](inputs))
        # temp = torch.max(a.data, 1)
        # print(temp[1], a)
        #
        # bb = model_0(inputs)
        # print (bb)
        # b = softmax(model_1(bb))
        # temp = torch.max(b.data, 1)
        # print(temp[1], b)

        # 计数
        total += labels.size(0)
        # 按照退出点依次计算
        start_time = time.time()
        for i in range(model_num):
            # 计算中间结果和退出点输出
            output_temp[i] = model_list[2*i](inputs)
            if i < model_num-1:
                output = model_list[2*i + 1](output_temp[i])
            else:
                # 最后一个出口，直接计算
                output = output_temp[i]
            output_list[i] = output
            inputs = output_temp[i]
            # 计算阈值
            Threshold[i] = get_threshold(output)
            # 退出条件判断
            if Threshold[i] > exit_threshold[i]:
                # 满足退出条件
                end_time = time.time()
                exit_count[i] += 1
                softmax = nn.Softmax(dim=1)
                # _, predicted = torch.max(softmax(output.data), 1)
                _, predicted = torch.max(output.data, 1)
                pre_right_count[i] += (predicted == labels).sum().item()
                run_time[i] += (end_time - start_time)
                break

            elif i == (model_num - 1):
                # 如果到主网络依然不满足退出条件
                # 需要多个网络投票，决出最终结果，新的决策方式
                # print ("Voting Network")
                end_time = time.time()
                voting_count += 1
                # 此函数可能存在问题，需后期调试##########################################
                voting_softmax = get_voting_softmax(output_list, voting_weight)
                _, predicted = torch.max(voting_softmax.data, 1)
                # if torch.equal(labels, predicted):
                #     # print ("相等")
                #     voting_right_count += 1
                voting_right_count += (predicted == labels).sum().item()
                # other_prediction_right += (predicted == labels).sum().item()
                voting_time += (end_time - start_time)
                break
            else:
                continue

    temp_end_time = time.time()
    # 打印测试信息
    print ('总共测试 %d条数据' % total)
    for i in range(model_num):
        if exit_count[i]==0:
            print ("Exit point: %d，Count: %d" % (i, exit_count[i]))
        else:
            print("Exit point: %d，Count: %d, Pre_acc: %2.2f" % ( i, exit_count[i], pre_right_count[i] / exit_count[i] * 100), "%",
                  " Threshold: %.3f, one_used_time:%.3fms, Exit Percentage: %2.2f" % (exit_threshold[i], run_time[i] / exit_count[i] * 1000, exit_count[i] / total * 100), "%")
    if voting_count==0:
        print ("Voting count: %d," % voting_count)
    else:
        print ("Voting count: %d, Pre_acc: %2.2f" % (voting_count, voting_right_count / voting_count * 100), "%","one_used_time:%.3fms, Percentage: %2.2f" % (voting_time / voting_count * 1000, voting_count / total * 100), "%")
    total_acc = (sum(pre_right_count) + voting_right_count)/total*100
    one_used_time = (sum(run_time) + voting_time)/total*1000
    print ("Total Pre_acc:%2.2f" % (total_acc), "%", "one_used_time:%.3fms" % (one_used_time))
    total_time = (temp_end_time - temp_start_time)/total*1000
    print("Total Pre_acc:%2.2f" % (total_acc), "%", "one_used_time:%.3fms" % (total_time) )
    print ("Eval BranchyNet 结束")



def test():
    model_1, model_2, model_3, model_4, model_5 = get_inference_model.get_model(num_classes=100)

    # for i, data in enumerate(model_5.named_parameters()):
    #     name, param = data
    #     print (i, name, param.size())
    model_num = 3
    Project_dir = my_functions.get_project_dir()
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)

    for i in range(model_num):
        # print (i)
        dir = Project_dir + "/model/ResNet/KD/ResNet_As_model_" + str(i) + "_checkpoint.tar"
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
        branch_model[i].load_state_dict(checkpoint["state_dict"])

    # 初始化
    copy_layer_param(branch_model[0], model_1, [0, 29])
    copy_layer_param(branch_model[0], model_2, [30, 48])
    copy_layer_param(branch_model[1], model_3, [30, 44])
    copy_layer_param(branch_model[1], model_4, [45, 57])
    copy_layer_param(branch_model[2], model_5, [45, 61])



    train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 1)
    num = 1
    start_time = time.time()
    model_1.eval()
    model_2.eval()
    for i in range(model_num):
        branch_model[i].eval()

    # for i, data in enumerate(model_2.named_parameters()):
    #     name, param = data
    #     print (i, name, param)

    # for i, data in enumerate(branch_model[0].named_parameters()):
    #     name, param = data
    #     print (i, name, param)

    for i, data in enumerate(test_loader, 0):
        if i < num:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            out = model_1(inputs)
            # out = model_2(out)
            out = model_3(out)
            # print (out.size())
            print (out)
            out = model_4(out)
            # out = model_5(out)
            print(out)
            new_out = branch_model[1](inputs)
            print(new_out)
    end_time = time.time()
    print("time : %2.2d s" % (end_time - start_time))


    start_time = time.time()
    for i, data in enumerate(test_loader, 0):
        if i < num:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            out = branch_model[1](inputs)
            # print (out)
    end_time = time.time()
    print("time : %2.2d s" % (end_time - start_time))


if __name__ == "__main__":

    Eval_Branch_model()
    # 创建网络连接
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.connect(('192.168.10.128', 5000))
    # 直接调用socket通信

    # model_1_out = 0
    # model_3_out = 0
    # col_com()


    # 结束
    # msg = "bye"
    # sock.send(str.encode(msg))
    # sock.close()

    # main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    # model = branch_model[2]
    # model_0 = ResNet_0(BasicBlock, [2,2,2,2],num_classes=100)
    # model_1 = ResNet_1(BasicBlock, [2,2,2,2],num_classes=100)
    #
    # dir = my_functions.get_project_dir() + "/model/ResNet/KD/ResNet_As_model_2_checkpoint.tar"
    # checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint["state_dict"])
    #
    # copy_layer_param(model, model_0, [0, 44])
    # # print (model.state_dict())
    # # print (model_0.state_dict())
    # copy_layer_param(model, model_1, [45, 61])
    # # print (model_1.state_dict())
    #
    # # print (model_0)
    # # summary(model_0, input_size=(3, 32, 32))
    # # summary(model_1, input_size=(256, 8, 8))
    # train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=1)
    #
    # num =100
    # start_time = time.time()
    # for i, data in enumerate(test_loader, 0):
    #     if i < num:
    #         # print (i)
    #         inputs, labels = data
    #         inputs, labels = Variable(inputs), Variable(labels)
    #
    #         middle_output = model_0(inputs)
    #         divide_output = model_1(middle_output)
    #         # print(divide_output)
    #     else:
    #         break
    # end_time = time.time()
    # print("time : %.2d s" % (end_time - start_time) )
    #
    # start_time = time.time()
    # for i, data in enumerate(test_loader, 0):
    #     if i < num:
    #         # print (i)
    #         inputs, labels = data
    #         inputs, labels = Variable(inputs), Variable(labels)
    #
    #         output = model(inputs)
    #         # print("time : ", end_time - start_time)
    #         # print (output)
    #     else:
    #         break
    # end_time = time.time()
    # print("time : %.2d s" % (end_time - start_time))

# train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 1)
# main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
# model = branch_model[2]

# my_model = nn.Sequential(*list(model.children())[1:5])
# print (my_model)
# for idx, m in enumerate(model.modules()):
#     key ,value = m
#     print(idx, '->', key)