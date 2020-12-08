# 测试某些功能


import torch
from torch import nn
import time
from functions import my_functions
from model.ResNet import get_ResNet_model, get_inference_model
from functions.branch_functions import Train_BranchyNet_Asynchronous, Train_BranchyNet_Synchronization, Eval_BranchyNet, \
    get_Exit_Threshold, Train_BranchyNet_Asynchronous_KD, Train_BranchyNet_Asynchronous_KD_Back, Train_BranchyNet_Asynchronous_Back, \
    get_threshold, get_voting_softmax
from datasets import get_cifar_100

# def Eval_Branch_model(exit_threshold = [0.95,0.97,0.98], voting_weight = [0.30, 0.30, 0.40]):
def Eval_Branch_model():

    exit_threshold = [0.96, 0.97, 0.98]
    voting_weight = [0.30, 0.30, 0.40]
    test_num = 9990
    model_num = 3
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
    # 设置为eval模式
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    # 将model设置为eval模式，防止模型参数变化
    for i in range(model_num):
        branch_model[i].eval()
    # GPU资源能否用
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

    for j in range(5):
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
                    # print ("Voting Network")
                    end_time = time.time()
                    voting_count += 1
                    # 此函数可能存在问题，需后期调试##########################################
                    voting_softmax = get_voting_softmax(output_list, voting_weight)
                    _, predicted = torch.max(voting_softmax.data, 1)
                    voting_right_count += (predicted == labels).sum().item()
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
        # print("Total Pre_acc:%2.2f" % (total_acc), "%", "one_used_time:%.3fms" % (total_time) )
        print ("Eval BranchyNet Time 结束")

if __name__ == "__main__":
    Eval_Branch_model()
