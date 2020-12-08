# 将祖目录加入临时路径
import sys
sys.path.append("../..")
sys.path.append("..")
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from datasets import get_cifar_100
import torch.optim as optim
from functions.my_functions import Eval_model, Train_model, copy_layer_param, get_project_dir, adjust_learning_rate, fine_tune_adjust_learning_rate, \
    get_JS_divergence, get_KL_divergence, get_Wasserstein_distance

# 由softmax获得阈值
def get_threshold(outputs):
    # Method 1
    softmax = nn.Softmax(dim=1)
    result = softmax(outputs)
    temp = torch.max(result.data, 1)
    temp = temp[0].cpu().numpy()
    return temp[0]

# 加权投票
def get_voting_softmax(output_list, voting_weight):
    '''
    :param output_list: 各个退出点的输出
    :param voting_weight: 投票权重
    :return: 投票结果
    '''
    softmax = nn.Softmax(dim=1)
    length = len(output_list)
    voting_softmax = output_list[0]*0
    # for i in range(len(output_list[0])):
    #     voting_softmax.append(0.0)
    for i in range(length):
        voting_softmax += softmax(output_list[i]) * voting_weight[i]
    return voting_softmax

# 获得退出阈值
def get_Exit_Threshold(branch_model, test_loader, proportion):
    Exit_Threshold = []
    model_num = len(branch_model)
    for i in range(model_num):
        branch_model[i].eval()
        Threshold = []
        total = 0
        for key, data in enumerate(test_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            total += labels.size(0)
            output = branch_model[i](inputs)
            Threshold.append(get_threshold(output))
        Threshold.sort(reverse=True)
        flag = int(total * proportion[i])
        Exit_Threshold.append(Threshold[flag])
    print (Exit_Threshold)
    return Exit_Threshold

# 获得损失值
def get_loss(outputs, labels):
    # 定义每一个exit的权重
    loss_weight = [0.2, 0.2, 0.6]
    criterion = nn.CrossEntropyLoss()
    length = len(outputs)
    # loss_weight_total = criterion(outputs[0], labels)
    # loss_weight_total = loss_weight_total - loss_weight_total
    loss_weight_total = criterion(outputs[0], labels) * 0.0
    loss_list = []
    total_loss = []
    # 计算单个分支的损失函数
    for i in range(length):
        loss_list.append( criterion(outputs[i], labels) )
    # 初始化总损失
    for i in range(length):
        loss_weight_total += loss_list[i]*loss_weight[i]
    # 计算总的损失
    for i in range(length):
        total_loss.append(loss_weight_total)
    return total_loss, loss_list

# 测试 BranchyNet的性能
def Eval_BranchyNet(branch_model, exit_threshold, voting_weight, test_loader, use_cuda=True):
    '''
    :param branch_model: 分支网络
    :param exit_threshold: 各分支退出阈值设定
    :param test_loader: 数据测试集, batch_size = 1
    :return: 无
    注意模型是 GPU 还是 cpu
    '''
    model_num = len(branch_model)
    # 单网络测试
    # for i in range(model_num):
    #     pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
    #     # print info
    #     print("Eval info, BranchyNet[%d]，Acc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
    #           % (i, pre_acc, loss, total_time, once_time * 1000))

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

    # 将model设置为eval模式，防止模型参数变化
    for i in range(model_num):
        branch_model[i].eval()

    temp_start_time = time.time()
    # 开始测试BranchyNet
    for key, data in enumerate(test_loader):
        if key == 500:
            break
        Threshold = []
        output_list = []
        for i in range(model_num):
            Threshold.append(0.0)
            output_list.append(0.0)

        # start_time = time.time()
        inputs, labels = data
        # 使用cuda，并且有cuda资源
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # 计数
        total += labels.size(0)
        # 按照退出点依次计算
        for i in range(model_num):
            # 新的计时位置
            start_time = time.time()
            output = branch_model[i](inputs)
            output_list[i] = output
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
    total_time = (temp_end_time - temp_start_time)/10000*1000
    print("Total Pre_acc:%2.2f" % (total_acc), "%", "one_used_time:%.3fms" % (total_time) )
    print ("Eval BranchyNet 结束")

# 同步训练 BranchyNet，
def Train_BranchyNet_Synchronization(branch_model, Epoch = 300, copy_range=[-1], DataSet="cifar-100", Print_epoch_fre = 1, save_dir="LeNet"):

    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 256)
    elif DataSet == "mnist":
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size = 128, test_batch_size = 256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size = 128, test_batch_size = 256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    # get model and init it
    model_num = len(branch_model)
    # load over
    Project_dir = get_project_dir()
    dir = Project_dir + "/model/" + save_dir + "sy/"

    # define train param
    learning_rate = 0.1
    # define model_num optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    op = []
    for i in range(model_num):
        # 定义优化器，
        # op.append(optim.Adam( branch_model[i].parameters(), lr = learning_rate))
        op.append(optim.SGD(branch_model[i].parameters(), lr=learning_rate, momentum=0.9))
        # load to GPU
        if torch.cuda.is_available():
            branch_model[i] = branch_model[i].cuda()

    # 定义指标，记录过程
    best_acc = 0.0
    best_acc_list = []
    train_total = 0
    best_state_dict = {}
    # 长度为model_num,针对每个网络，数值一样
    train_total_loss = []
    correct = []
    total_loss = []
    loss_list = []
    # 预测准确率
    pre_acc_list = []
    for i in range(model_num):
        train_total_loss.append(0.0)
        correct.append(0.0)
        pre_acc_list.append(0.0)
        best_acc_list.append(0.0)
        total_loss.append(0.0)
        loss_list.append(0.0)

    for epoch in range(Epoch):
        start_time = time.time()
        for key, data in enumerate(train_loader, 0):
            # 调节学习率
            for i in range(model_num):
                adjust_learning_rate(op[i], epoch)
            # get inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            for i in range(model_num):
                op[i].zero_grad()
            outputs = []
            # 对每个网络进行循环，得到输出
            for i in range(model_num):
                outputs.append(branch_model[i](inputs))
            # loss为加权后的损失函数，train_loss_temp为零时list
            total_loss, loss_list = get_loss(outputs, labels)
            # 反向传递
            # 计算反向传播loss.backward(),##############################################################################
            for i in range(model_num-1, -1, -1):
                if i != 0:
                    total_loss[i].backward(retain_graph = True)
                else:
                    total_loss[i].backward()
                op[i].step()
            # 神经网络参数复制, 复制 main_model 参数
            for i in range(model_num-2, -1, -1):
                # 遍历，从第一层复制到第n层
                # 参数不需要固定，每个循环后更新参数
                copy_layer_param(branch_model[model_num - 1], branch_model[i], copy_range[i])
            # 计数、时间及其他指标
            train_total += labels.size(0)
            for i in range(model_num):
                _, predicted = torch.max(outputs[i].data, 1)
                correct[i] += (predicted == labels).sum().item()
                train_total_loss[i] += loss_list[i].item() * labels.size(0)
        end_time = time.time()
        # 打印训练信息
        train_acc =  []
        for i in range(model_num):
            train_acc.append(correct[i]/train_total)
            # print ("Epoch %d, Branchy[%d], train_acc:%2.2f" % (epoch, i, correct[i]/train_total),"%",
            #        " loss: %3.6f ,Training used time: %1.2fs " % ( train_total_loss[i]/train_total, end_time-start_time))
        print ("Epoch:%d, Training used time: %.1fs" % (epoch, end_time-start_time), "Train acc list:", train_acc, "Train loss list:", loss_list)
        # 打印测试信息
        # 测试各个网络的的精确度，及运行时间,依次打印
        total_pre_acc = 0.0
        for i in range(model_num):
            pre_acc_list[i], loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
            # print info
            if ( epoch % Print_epoch_fre ) == 0:
                print ("BranchyNet[%d] Eval info,Prediction Acc:%2.2f" % (i, pre_acc_list[i]), "%",
                       "Loss:%3.4f, used_time:%.2fs, one_used_time:%.4fms" % (loss, total_time, once_time*1000))
        print ("\n")
        total_pre_acc = sum(pre_acc_list)/len(pre_acc_list)
        if best_acc < total_pre_acc:
            best_acc = total_pre_acc
            best_acc_list = pre_acc_list
            # 保存精度最好的网络
            for i in range(model_num):
                best_state_dict[i] = branch_model[i].state_dict()
    # 重新将做好的网络模型参数返回模型
    for i in range(model_num):
        branch_model[i].load_state_dict(best_state_dict[i])
    print ("Best prediction Acc :%2.2f" % best_acc, "%", "Best Prediction List: ", best_acc_list)
    # save model
    # print ("Save model ")
    for i in range(model_num):
        torch.save({
            "best_acc": best_acc,
            "state_dict": branch_model[i].state_dict(),
        }, dir + "BranchyNet[" + str(i) + "]_sy_checkpoint.tar")
    print ("Finished BranchyNet Synchronization Training ")
    # 返回整个模型
    return branch_model


# 异步训练前向BranchyNet
def Train_BranchyNet_Asynchronous(branch_model, Epoch_list = [-1], copy_range=[-1], DataSet="cifar-100", Print_epoch_fre = 1, save_dir="LeNet"):

    model_num = len(branch_model)

    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 256)
    elif DataSet == "mnist":
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size = 128, test_batch_size = 256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size = 128, test_batch_size = 256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)

    Project_dir = get_project_dir()
    dir = Project_dir + "/model/temp_model/"
    # define train param
    learning_rate = 0.1
    # define model_num optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    op = []
    for i in range(model_num):
        # 定义优化器,确定参数更新范围
        if i == model_num-1:
            op.append(optim.SGD(branch_model[i].parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
        else:
            op.append(optim.SGD(filter(lambda p: p.requires_grad, branch_model[i].parameters()), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
    # load to GPU
    for i in range(model_num):
        if torch.cuda.is_available():
            branch_model[i] = branch_model[i].cuda()

    # 初始化参数
    start_time = []
    end_time = []
    used_time = []
    train_total = []
    train_total_loss = []
    correct = []
    # 记录值初始化长度与数值
    for x in range(model_num):
        start_time.append(0.0)
        end_time.append(0.0)
        used_time.append(0.0)
        train_total.append(0)
        train_total_loss.append(0.0)
        correct.append(0)

    # 先训练主网络，再训练分支网络
    print ("Train_BranchyNet_Asynchronous Begin")
    for i in range(model_num-1, -1, -1):
        # Firstly Train main model
        print ("Train %d model" % i)
        if (i == model_num - 1):
            # 训练 主网络， BranchyNet main_model
            best_acc = 0.0
            best_state_dict = {}
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)
                    criterion = nn.CrossEntropyLoss()
                    loss =  criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # 不需要冻结部分参数
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()

                # 对main_model模型进行测试
                end_time[i] = time.time()
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Training Used time:%2.2fs, Prediction Acc:%2.2f" % (epoch, end_time[i]-start_time[i], pre_acc),"%",
                          ", loss: %2.5f, Eval used time: %.2fs, one used time: %.4fms" % (loss, total_time, once_time*1000))
                # 存储准确率最好的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载已保存的最好参数
            branch_model[i].load_state_dict(best_state_dict)
            print ("Best prediction Acc :%2.2f" % best_acc, "%")
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_checkpoint.tar")
            # 主网络训练完成 #################################################################################################################################


        else: # 训练分支网络
            # 训练分支网络，需要初始化参数，且固定住。
            print ("训练分支网络：", i)
            best_acc = 0.0
            best_state_dict = {}
            # 初始化分支网络， 将 网络重合部分 初始化，branch_model[i],复制branch_model[num_model-1]的参数
            copy_layer_param(branch_model[model_num - 1], branch_model[i], copy_range[i])
            # 定义优化器，设置False，确定那些layer参数不参与更新
            for index, param in enumerate(branch_model[i].parameters()):
                if index <= copy_range[i]:
                    param.requires_grad = False
            # 开始训练网络
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()

                fine_tune_adjust_learning_rate(op[i], epoch)

                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)
                    loss =  criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # layer重合部分已设置为False，不会更新
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()
                # 对main_model模型进行测试
                end_time[i] = time.time()
                used_time[i] = end_time[i] - start_time[i]
                # print("Epoch %3d, train_acc:%0.4f, loss: %3.6f ,Training used time: %1.2f s "
                #       % (epoch, correct[i] / train_total[i], train_total_loss[i] / train_total[i], end_time[i] - start_time[i]))
                # eval_model
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Used time:%2.2fs, Prediction Acc:%2.2f" % (epoch, end_time[i] - start_time[i], pre_acc),
                          "%",", loss: %2.5f, eval used time: %.2fs, one used time: %.4fms" % ( loss, total_time, once_time * 1000))
                # 存储准确率最高的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载准确率最高的网络参数
            branch_model[i].load_state_dict(best_state_dict)
            print("Best prediction Acc :%2.2f" % best_acc, "%")
            # 保存网络模型
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_as_checkpoint.tar")
    # 训练结束
    print("Finished BranchyNet Asynchronous Training ")
    return branch_model

# 异步反向训练 BranchyNet， 3 -> 2 -> 1
def Train_BranchyNet_Asynchronous_Back(branch_model, Epoch_list = [-1], copy_range=[-1], DataSet="cifar-100", Print_epoch_fre = 1, save_dir="LeNet"):

    model_num = len(branch_model)

    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 256)
    elif DataSet == "mnist":
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size = 128, test_batch_size = 256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size = 128, test_batch_size = 256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    Project_dir = get_project_dir()
    dir = Project_dir + "/model/temp_model/"
    # define train param
    learning_rate = 0.1
    # define model_num optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    op = []
    for i in range(model_num):
        # 定义优化器,确定参数更新范围
        # op.append(optim.Adam(filter(lambda p: p.requires_grad, branch_model[i].parameters()), lr=learning_rate))
        if i == 0:
            # 主网络
            # op.append(optim.Adam(branch_model[i].parameters(), lr=learning_rate))
            op.append(optim.SGD(branch_model[i].parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
        else:
            # 分支网络
            # op.append(optim.Adam(filter(lambda p: p.requires_grad, branch_model[i].parameters()), lr=learning_rate))
            op.append(optim.SGD(filter(lambda p: p.requires_grad, branch_model[i].parameters()), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
    # load to GPU
    for i in range(model_num):
        if torch.cuda.is_available():
            branch_model[i] = branch_model[i].cuda()

    # 初始化参数
    start_time = []
    end_time = []
    used_time = []
    train_total = []
    train_total_loss = []
    correct = []
    # 记录值初始化长度与数值
    for x in range(model_num):
        start_time.append(0.0)
        end_time.append(0.0)
        used_time.append(0.0)
        train_total.append(0)
        train_total_loss.append(0.0)
        correct.append(0)

    # 先训练主网络，再训练分支网络
    print ("Train_BranchyNet_Asynchronous_Back Begin")
    for i in range(1, model_num):

        # Train main model
        print ("Train %d model" % i)
        if (i == 0):
            # 训练 主网络， BranchyNet main_model
            best_acc = 0.0
            best_state_dict = {}
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)
                    criterion = nn.CrossEntropyLoss()
                    loss =  criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # 不需要冻结部分参数
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()

                # 对main_model模型进行测试
                end_time[i] = time.time()
                # print("Epoch:%3d, train_acc:%0.4f, loss:%2.6f ,Training used time:%1.2f s "
                #       % (epoch, correct[i]/train_total[i], train_total_loss[i] / train_total[i], end_time[i] - start_time[i]))
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Training Used time:%2.2fs, Prediction Acc:%2.2f" % (epoch, end_time[i]-start_time[i], pre_acc),"%",
                          ", loss: %2.5f, Eval used time: %.2fs, one used time: %.4fms" % (loss, total_time, once_time*1000))
                # 存储准确率最好的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载已保存的最好参数
            branch_model[i].load_state_dict(best_state_dict)
            print ("Best prediction Acc :%2.2f" % best_acc, "%")
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_checkpoint.tar")
            # 主网络训练完成 #################################################################################################################################


        else: # 训练分支网络
            # 训练分支网络，需要初始化参数，且固定住。
            print ("训练分支网络：", i)
            best_acc = 0.0
            best_state_dict = {}
            # 初始化分支网络， 将 网络重合部分 初始化，branch_model[i],复制branch_model[num_model-1]的参数
            # copy_layer_param(branch_model[model_num - 1], branch_model[i], copy_range[i])
            copy_layer_param(branch_model[i - 1], branch_model[i], copy_range[i])
            # 定义优化器，设置False，确定那些layer参数不参与更新
            for index, param in enumerate(branch_model[i].parameters()):
                if index <= copy_range[i]:
                    param.requires_grad = False
            # 开始训练网络
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                # adjust_learning_rate(op[i], epoch)
                fine_tune_adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)
                    loss =  criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # layer重合部分已设置为False，不会更新
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()
                # 对main_model模型进行测试
                end_time[i] = time.time()
                used_time[i] = end_time[i] - start_time[i]
                # eval_model
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Used time:%2.2fs, Prediction Acc:%2.2f" % (epoch, end_time[i] - start_time[i], pre_acc),
                          "%",", loss: %2.5f, eval used time: %.2fs, one used time: %.4fms" % ( loss, total_time, once_time * 1000))
                # 存储准确率最高的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载准确率最高的网络参数
            branch_model[i].load_state_dict(best_state_dict)
            print("Best prediction Acc :%2.2f" % best_acc, "%")
            # 保存网络模型
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_as_checkpoint.tar")
    # 训练结束
    print("Finished BranchyNet Asynchronous Training ")
    return branch_model


# KD异步前向训练BranchyNet
def Train_BranchyNet_Asynchronous_KD(branch_model, teacher_model, Epoch_list = [-1], copy_range=[-1], DataSet="cifar-100",
                                     Print_epoch_fre = 1, Distance_type="KL", Temperature=4.0, beta=0.1):

    model_num = len(branch_model)
    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    elif DataSet == "mnist":
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size=128, test_batch_size=256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size=128, test_batch_size=256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)

    Project_dir = get_project_dir()
    dir = Project_dir + "/model/temp_model/"
    # define train param
    learning_rate = 0.1
    # define model_num optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    my_softmax = nn.Softmax(dim=1)
    op = []
    for i in range(model_num):
        # 定义优化器,确定参数更新范围
        if i == model_num - 1:
            op.append(optim.SGD(branch_model[i].parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
        else:
            # 分支网络
            op.append(optim.SGD(filter(lambda p: p.requires_grad, branch_model[i].parameters()), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
    # load to GPU
    for i in range(model_num):
        if torch.cuda.is_available():
            branch_model[i] = branch_model[i].cuda()
    # 初始化参数
    start_time = []
    end_time = []
    used_time = []
    train_total = []
    train_total_loss = []
    correct = []
    # 记录值初始化长度与数值
    for x in range(model_num):
        start_time.append(0.0)
        end_time.append(0.0)
        used_time.append(0.0)
        train_total.append(0)
        train_total_loss.append(0.0)
        correct.append(0)

    # 先训练主网络，再训练分支网络
    print("Train_BranchyNet_Asynchronous Begin")
    # 从 小 到 大
    for i in range(model_num - 1, -1, -1):
        # Train main model
        print("Train %d model" % i)
        if (i == model_num-1):
            # 训练 主网络， BranchyNet main_model
            best_acc = 0.0
            best_state_dict = {}
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)
                    outputs_T = outputs / Temperature
                    outputs_softmax = my_softmax(outputs_T)
                    teacher_outputs = teacher_model(inputs) / Temperature
                    teacher_outputs_softmax = my_softmax(teacher_outputs)

                    if Distance_type == "JS":
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_JS_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "WS":
                        # loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_Wasserstein_distance(
                        #     teacher_outputs_softmax, outputs_softmax)
                        loss = criterion(outputs, labels) +  2 * Temperature * Temperature * get_Wasserstein_distance(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "KL":
                        # 默认为KL散度
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature* get_KL_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # 不需要冻结部分参数
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()
                # 对main_model模型进行测试
                end_time[i] = time.time()
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Training Used time:%2.2fs, Prediction Acc:%2.2f" % (epoch, end_time[i] - start_time[i], pre_acc), "%",
                          ", loss: %2.5f, Eval used time: %.2fs, one used time: %.4fms" % (loss, total_time, once_time * 1000))
                # 存储准确率最好的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载已保存的最好参数
            branch_model[i].load_state_dict(best_state_dict)
            print("Best prediction Acc :%2.2f" % best_acc, "%")
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_checkpoint.tar")

        else:  # 训练分支网络
            # 训练分支网络，需要初始化参数，且固定住。
            print("训练分支网络：", i)
            best_acc = 0.0
            best_state_dict = {}
            # 初始化分支网络， 将 网络重合部分 初始化，branch_model[i],复制branch_model[num_model-1]的参数
            copy_layer_param(branch_model[model_num - 1], branch_model[i], copy_range[i])
            # 定义优化器，设置False，确定那些layer参数不参与更新
            for index, param in enumerate(branch_model[i].parameters()):
                if index <= copy_range[i]:
                    param.requires_grad = False
            # 开始训练网络
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                # adjust_learning_rate(op[i], epoch)
                fine_tune_adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)/ Temperature
                    outputs_softmax = my_softmax(outputs)

                    teacher_outputs = teacher_model(inputs) / Temperature
                    teacher_outputs_softmax = my_softmax(teacher_outputs)

                    if Distance_type == "JS":
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_JS_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "WS":
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_Wasserstein_distance(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "KL":
                        # 默认为KL散度
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature* get_KL_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # layer重合部分已设置为False，不会更新
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()
                # 对main_model模型进行测试
                end_time[i] = time.time()
                used_time[i] = end_time[i] - start_time[i]
                # eval_model
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Used time:%2.2fs, Prediction Acc:%2.2f" % (epoch, end_time[i] - start_time[i], pre_acc),
                          "%", ", loss: %2.5f, eval used time: %.2fs, one used time: %.4fms" % (loss, total_time, once_time * 1000))
                # 存储准确率最高的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载准确率最高的网络参数
            branch_model[i].load_state_dict(best_state_dict)
            print("Best prediction Acc :%2.2f" % best_acc, "%")
            # 保存网络模型
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_as_checkpoint.tar")
    # 训练结束
    print("Finished BranchyNet Asynchronous Training ")
    return branch_model

# KD异步反向训练BranchyNet
def Train_BranchyNet_Asynchronous_KD_Back(branch_model, teacher_model, Epoch_list = [-1], copy_range=[-1], DataSet="cifar-100",
                                     Print_epoch_fre = 1, Distance_type="KL", Temperature=4.0, beta=0.1):

    model_num = len(branch_model)
    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    elif DataSet == "mnist":
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size=128, test_batch_size=256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size=128, test_batch_size=256)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)

    Project_dir = get_project_dir()
    dir = Project_dir + "/model/temp_model/"
    # define train param
    learning_rate = 0.1
    # define model_num optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    my_softmax = nn.Softmax(dim=1)
    op = []
    for i in range(model_num):
        if i == 0:
            op.append(optim.SGD(branch_model[i].parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4))
        else:
            op.append(optim.SGD(filter(lambda p: p.requires_grad, branch_model[i].parameters()), lr=learning_rate,
                                momentum=0.9, weight_decay=5e-4))
    # load to GPU
    for i in range(model_num):
        if torch.cuda.is_available():
            branch_model[i] = branch_model[i].cuda()
    # 初始化参数
    start_time = []
    end_time = []
    used_time = []
    train_total = []
    train_total_loss = []
    correct = []
    # 记录值初始化长度与数值
    for x in range(model_num):
        start_time.append(0.0)
        end_time.append(0.0)
        used_time.append(0.0)
        train_total.append(0)
        train_total_loss.append(0.0)
        correct.append(0)

    # 先训练主网络，再训练分支网络
    print("Train_BranchyNet_Asynchronous Begin")
    for i in range(model_num - 1, -1, -1):
    # for i in range(model_num):
        # Train main model
        print("Train %d model" % i)
        if (i == 0):
            # 训练 主网络， BranchyNet main_model
            best_acc = 0.0
            best_state_dict = {}
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)
                    outputs_T = outputs / Temperature
                    outputs_softmax = my_softmax(outputs_T)
                    teacher_outputs = teacher_model(inputs) / Temperature
                    teacher_outputs_softmax = my_softmax(teacher_outputs)

                    if Distance_type == "JS":
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_JS_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "WS":
                        # loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_Wasserstein_distance(
                        #     teacher_outputs_softmax, outputs_softmax)
                        loss = criterion(outputs, labels) + 2 * Temperature * Temperature * get_Wasserstein_distance(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "KL":
                        # 默认为KL散度
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature* get_KL_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # 不需要冻结部分参数
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()

                # 对main_model模型进行测试
                end_time[i] = time.time()
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Training Used time:%2.2fs, Prediction Acc:%2.2f" % (
                    epoch, end_time[i] - start_time[i], pre_acc), "%",
                          ", loss: %2.5f, Eval used time: %.2fs, one used time: %.4fms" % (
                          loss, total_time, once_time * 1000))
                # 存储准确率最好的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载已保存的最好参数
            branch_model[i].load_state_dict(best_state_dict)
            print("Best prediction Acc :%2.2f" % best_acc, "%")
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_checkpoint.tar")

        else:  # 训练分支网络
            # 训练分支网络，需要初始化参数，且固定住。
            print("训练分支网络：", i)
            best_acc = 0.0
            best_state_dict = {}
            # 初始化分支网络， 将 网络重合部分 初始化，branch_model[i],复制branch_model[num_model-1]的参数
            copy_layer_param(branch_model[i - 1], branch_model[i], copy_range[i])
            # copy_layer_param(branch_model[model_num - 1], branch_model[i], copy_range[i])
            # 定义优化器，设置False，确定那些layer参数不参与更新
            for index, param in enumerate(branch_model[i].parameters()):
                if index <= copy_range[i]:
                    param.requires_grad = False
            # 开始训练网络
            for epoch in range(Epoch_list[i]):
                start_time[i] = time.time()
                # adjust_learning_rate(op[i], epoch)
                fine_tune_adjust_learning_rate(op[i], epoch)
                for key, data in enumerate(train_loader, 0):
                    # get inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs, labels = Variable(inputs), Variable(labels)
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    # 对每个网络进行循环，得到输出
                    op[i].zero_grad()
                    outputs = branch_model[i](inputs)/ Temperature
                    outputs_T = outputs /Temperature
                    outputs_softmax = my_softmax(outputs_T)

                    teacher_outputs = teacher_model(inputs) / Temperature
                    teacher_outputs_softmax = my_softmax(teacher_outputs)

                    if Distance_type == "JS":
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_JS_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "WS":
                        # loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature * get_Wasserstein_distance(
                        #     teacher_outputs_softmax, outputs_softmax)
                        loss = criterion(outputs, labels) + 2 * Temperature * Temperature * get_Wasserstein_distance(
                            teacher_outputs_softmax, outputs_softmax)
                    elif Distance_type == "KL":
                        # 默认为KL散度
                        loss = (1 - beta) * criterion(outputs, labels) + beta * Temperature * Temperature* get_KL_divergence(
                            teacher_outputs_softmax, outputs_softmax)
                    else:
                        loss = criterion(outputs, labels)
                    loss.backward()
                    op[i].step()
                    # layer重合部分已设置为False，不会更新
                    _, predicted = torch.max(outputs.data, 1)
                    correct[i] += (predicted == labels).sum().item()
                    train_total[i] += labels.size(0)
                    train_total_loss[i] += loss.item()
                # 对main_model模型进行测试
                end_time[i] = time.time()
                used_time[i] = end_time[i] - start_time[i]
                # eval_model
                pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
                if (epoch % Print_epoch_fre) == 0:
                    print("Epoch:%d, Used time:%2.2fs, Prediction Acc:%2.2f" % (
                    epoch, end_time[i] - start_time[i], pre_acc),
                          "%", ", loss: %2.5f, eval used time: %.2fs, one used time: %.4fms" % (
                          loss, total_time, once_time * 1000))
                # 存储准确率最高的网络参数
                if pre_acc > best_acc:
                    best_acc = pre_acc
                    best_state_dict = branch_model[i].state_dict()
            # 加载准确率最高的网络参数
            branch_model[i].load_state_dict(best_state_dict)
            print("Best prediction Acc :%2.2f" % best_acc, "%")
            # 保存网络模型
            torch.save({
                "best_acc": best_acc,
                "state_dict": branch_model[i].state_dict(),
            }, dir + "Branch_model[" + str(i) + "]_as_checkpoint.tar")
    # 训练结束
    print("Finished BranchyNet Asynchronous Training ")
    return branch_model
