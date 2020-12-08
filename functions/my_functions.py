# 将祖目录加入临时路径
import sys
sys.path.append("../..")
sys.path.append("..")
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from scipy.stats import wasserstein_distance
import scipy
import time
import os
from datasets import get_cifar_100
import torch.optim as optim

# nn Layer 参数复制
def copy_layer_param(old_model, new_model, layer_count):
    '''
    :param old_model:
    :param new_model:
    :param layer_count: 复制参数的最后一层，从 0 开始
    :return: 传地址，
    '''
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    # 遍历赋值
    for i ,data in enumerate(new_model.named_parameters()):
        name, param = data
        if i <= layer_count:
            new_state_dict[name] = old_state_dict[name]
        else:
            break
    new_model.load_state_dict(new_state_dict)

# 正常训练，调整学习率等参数
def adjust_learning_rate(optimizer, epoch):
    lr = 0.1
    if epoch < 100:
        lr = 0.1
    elif epoch < 200:
        lr = 0.01
    elif epoch < 320 :
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 微调训练，学习率参数调整
def fine_tune_adjust_learning_rate(optimizer, epoch):
    lr = 0.1
    if epoch < 80:
        lr = 0.1
    elif epoch < 180:
        lr = 0.01
    elif epoch < 240:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 计算两个张量间的wasserstein_distance
def get_Wasserstein_distance(tensor_A = [], tensor_B = []):
    A = tensor_A.data.cpu().numpy()
    B = tensor_B.data.cpu().numpy()
    if np.size(A) != np.size(B):
        print ('size not equal !')
        return
    else:
        loss = 0.0
        for i in range(np.size(A, 0)):
            loss += wasserstein_distance(A[i, :], B[i, :])
        loss = loss/np.size(A, 0)
    return  torch.tensor(loss)

def get_KL_divergence(tensor_A=[], tensor_B=[]):
    A = tensor_A.data.cpu().numpy()
    B = tensor_B.data.cpu().numpy()
    if np.size(A) != np.size(B):
        print ('size not equal !')
        return
    else:
        loss = 0.0
        for i in range(np.size(A, 0)):
            loss += scipy.stats.entropy(A[i, :], B[i, :])
        loss = loss/np.size(A, 0)
    return  torch.tensor(loss)

def get_JS_divergence(tensor_A = [], tensor_B = []):
    A = tensor_A.data.cpu().numpy()
    B = tensor_B.data.cpu().numpy()
    if np.size(A) != np.size(B):
        print ('size not equal !')
        return
    else:
        loss = 0.0
        for i in range(np.size(A, 0)):
            Middle = (A[i, :] + B[i, :]) / 2
            loss += (0.5*scipy.stats.entropy(A[i, :], Middle) + 0.5*scipy.stats.entropy(B[i, :], Middle))
        loss = loss/np.size(A, 0)
    return  torch.tensor(loss)

# 获得项目地址
def get_project_dir():
    # dir = os.path.split(os.path.realpath(__file__))[0]
    dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return dir

# 测试模型性能
# eval model,return pre_acc, loss, once_time
def Eval_model(model, test_loader, use_cuda = True):
    '''
    :param model: NN model
    :param test_loader: test data
    :return: pre_acc, loss, once_time
    '''
    total = 0
    total_loss = 0.0
    correct = 0
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    # 模式转换，防止参数变化
    model.eval()
    for i, data in enumerate(test_loader):
        # if i == 1000:
        #     break
        # get inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # 调用 GPU
        if torch.cuda.is_available() and use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += labels.size(0) * loss.item()
    # 计算 eval 指标
    end_time = time.time()
    pre_acc = 100 * correct / total
    loss = total_loss / total
    total_time = end_time - start_time
    once_time = (end_time - start_time) / total
    model.train()

    return pre_acc, loss, total_time, once_time

# 训练DNN模型
def Train_model(model, Temperature = 1.0, Epoch = 300, DataSet="cifar-10", Print_epoch_fre = 1, opti="SGD"):

    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 128)
    elif DataSet == "mnist":
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size = 128, test_batch_size = 128)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size = 128, test_batch_size = 128)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    # define training param
    learning_rate = 0.1
    best_acc = 0.0
    # define best record
    pre_acc_list = []
    best_state_dict = {}

    # define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    if opti == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        model = model.cuda()
    # begin training
    print("begin training model")
    for epoch in range(Epoch):
        # adjust learning rate
        if opti == "SGD":
            adjust_learning_rate(optimizer, epoch)
        running_loss = 0.0
        train_total = 0
        train_total_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimizer
            outputs = model(inputs)/Temperature
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()*labels.size(0)
            train_total += labels.size(0)
            running_loss += loss.item()
            fre = 20.0
            if (i % fre == (fre-1)):
                # print ("频次: ", i, "Loss : %.4f" %(running_loss/fre))
                # print("Epoch: %d Train num： %d, Loss : %.4f" % (epoch, i, running_loss / fre))
                running_loss = 0.0

        end_time = time.time()

        # eval model
        pre_acc, loss, total_time, once_time = Eval_model(model, test_loader)
        pre_acc_list.append(pre_acc)
        if (epoch % Print_epoch_fre) == 0:
            print("Epoch %3d, loss: %3.6f ,Training used time: %0.2fs " % ( epoch, train_total_loss / train_total, end_time - start_time))
            print("Prediction Acc: %2.2f" % pre_acc, "%", "test_loss: %.2f, eval used time: %.2f s, one used time %.4fms"
                  % (loss, total_time, once_time * 1000))
        if pre_acc > best_acc:
            best_acc = pre_acc
            best_state_dict = model.state_dict()
    model.load_state_dict(best_state_dict)
    print ("Prediction Acc List:", pre_acc_list)
    print ("Best prediction Acc: %2.2f" % (best_acc), "%")
    print('Finished Training')
    model.load_state_dict(best_state_dict)
    return model, best_acc

# 定义知识蒸馏过程
def Knowledge_distillation(Student_model, Teacher_model, Epoch=300, Temperature=4.0, beta= 0.1, DataSet="cifar-100",
                           Distance_type="KL", Print_epoch_fre = 1, copy_range=-1, opti="SGD"):

    if DataSet == "cifar-100":
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size = 128, test_batch_size = 128)
    elif DataSet == "mnist" :
        # train_loader, test_loader = get_minst_data.get_data(train_batch_size = 128, test_batch_size = 128)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)
    else:
        # train_loader, test_loader = get_cifar_10.get_data(train_batch_size = 128, test_batch_size = 128)
        train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=256)

    pre_acc_list = []
    best_state_dict = {}
    # define optimizer and loss function
    learning_rate = 0.1
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    my_softmax = nn.Softmax(dim=1)
    if opti == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, Student_model.parameters()), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, Student_model.parameters()), lr=0.001)
    for index, param in enumerate(Student_model.parameters()):
        if index <= copy_range:
            param.requires_grad = False

    if torch.cuda.is_available():
        Teacher_model = Teacher_model.cuda()
        Student_model = Student_model.cuda()

    # begin training
    print("begin training model")
    for epoch in range(Epoch):
        # adjust learning rate
        if opti == "SGD":
            adjust_learning_rate(optimizer, epoch)
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # get teacher model outputs
            teacher_outputs_T = Teacher_model(inputs) / Temperature
            # forward + backward + optimizer
            Student_outputs = Student_model(inputs)
            Student_outputs_T = Student_model(inputs) / Temperature

            teacher_outputs_softmax = my_softmax(teacher_outputs_T)
            Student_outputs_softmax = my_softmax(Student_outputs_T)

            # define new loss function + KL or JS or Wasserstein_distance
            # ###########  选择不同的损失函数，搭配不同的比例系数 ################################
            if Distance_type=="JS":
                loss = (1 - beta) * criterion(Student_outputs, labels) + beta * get_JS_divergence(
                    teacher_outputs_softmax, Student_outputs_softmax) * Temperature * Temperature
            elif Distance_type=="WS":
                # loss = (1 - beta) * criterion(outputs, labels) + beta * get_Wasserstein_distance(
                #     teacher_outputs_softmax, outputs_softmax) * Temperature * Temperature
                loss = criterion(Student_outputs, labels) + Temperature * Temperature * get_Wasserstein_distance(
                    teacher_outputs_softmax, Student_outputs_softmax)
            elif Distance_type=="KL":
                # 默认为KL散度
                loss = (1 - beta) * criterion(Student_outputs, labels) + beta * Temperature * Temperature * get_KL_divergence(
                    teacher_outputs_softmax, Student_outputs_softmax)
            else:
                loss = criterion(Student_outputs, labels)

            # 定义回传
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # fre = 200.0
            # if (i % fre == (fre - 1)):
            #     print ("频次： %d, Loss : %.4f" %(i, running_loss/fre))
            #     running_loss = 0.0
        end_time = time.time()

        # eval model
        pre_acc, loss, total_time, once_time = Eval_model(Student_model, test_loader)
        pre_acc_list.append(pre_acc)
        if (epoch % Print_epoch_fre) == 0:
            print("Epoch %d Training used time: %0.2fs" % (epoch, end_time - start_time))
            print("Prediction Acc: %2.2f" % pre_acc, "%", "test_loss: %.2f, eval used time: %.2f s, one used time %.4fms"
                  % (loss, total_time, once_time * 1000))
        if pre_acc > best_acc:
            best_acc = pre_acc
            best_state_dict = Student_model.state_dict()
    Student_model.load_state_dict(best_state_dict)
    print ("Prediction Acc List:", pre_acc_list)
    print("Best prediction Acc :%2.2f" % best_acc, "%")
    return Student_model, best_acc