# 将祖目录加入临时路径
import sys
sys.path.append("../..")
sys.path.append("..")

import torch
import time
from functions import my_functions
from model.ResNet import get_ResNet_model
from functions.my_functions import Train_model, Eval_model
from functions.my_functions import Knowledge_distillation
from functions.branch_functions import Eval_model, Eval_BranchyNet
from datasets import get_cifar_100

# 训练学生网络
def Train_ResNet():
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    # model_0 = branch_model[0]
    model_num = len(branch_model)
    for i in range(model_num):
        dir = Project_dir + "/model/ResNet/KD/ResNet_com_model_" + str(i) + "_checkpoint.tar"
        branch_model[i], best_acc = Train_model(branch_model[i], Temperature=1.0, Epoch=300, DataSet="cifar-100", Print_epoch_fre=1)
        torch.save({
            "best_acc": best_acc,
            "state_dict": branch_model[i].state_dict(),
        }, dir)

    # print("Train ResNet_0")
    # model_0, best_acc = Train_model(model_0, Temperature=1.0, Epoch= 250, DataSet="cifar-100", Print_epoch_fre=1)
    # dir = Project_dir + "/model/ResNet/KD/ResNet_model_0_checkpoint.tar"
    # torch.save({
    #     "best_acc": best_acc,
    #     "state_dict":model_0.state_dict(),
    # }, dir)
    #
    # print("Train ResNet_1")
    # model_1 = branch_model[1]
    # model_1, best_acc = Train_model(model_1, Temperature=1.0, Epoch=250, DataSet="cifar-100", Print_epoch_fre=1)
    # dir = Project_dir + "/model/ResNet/KD/ResNet_model_1_checkpoint.tar"
    # torch.save({
    #     "best_acc": best_acc,
    #     "state_dict": model_1.state_dict(),
    # }, dir)
    #
    # print("Train ResNet_2")
    # model_2 = branch_model[2]
    # model_2, best_acc = Train_model(model_2, Temperature=1.0, Epoch=250, DataSet="cifar-100", Print_epoch_fre=1)
    # dir = Project_dir + "/model/ResNet/KD/ResNet_model_2_checkpoint.tar"
    # torch.save({
    #     "best_acc": best_acc,
    #     "state_dict": model_2.state_dict(),
    # }, dir)

    print ("Train ResNet Over")

# 训练教师网络
def Train_Teacher(T = 1.0):
    print ("Train ResNet Teacher model, T=", T)
    model = get_ResNet_model.get_teacher_model(num_classes=100)
    Teacher_model, best_acc = Train_model(model, Temperature = T, Epoch = 300, DataSet = "cifar-100", Print_epoch_fre = 1)
    dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_" + str(int(T)) + "_checkpoint.tar"
    torch.save({
        "best_acc": best_acc,
        "state_dict": Teacher_model.state_dict(),
    }, dir)
    print("Train ResNet Teacher Over")

# 知识蒸馏过程
def KD_ResNet(T = 1.0, Epoch = 400, Distance_type = "KL"):
    # 获得教师和学生模型
    print ("KD_ResNet, Temperature=", T, "Distance_type=", Distance_type)
    # Teacher_dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_" + str(int(T)) + "_checkpoint.tar"
    Teacher_dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_1_checkpoint.tar"
    Teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)
    checkpoint = torch.load(Teacher_dir, map_location=lambda storage, loc: storage)
    Teacher_model.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.is_available():
        Teacher_model = Teacher_model.cuda()

    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    model_num = len(branch_model)
    for i in range(model_num):
        dir = Project_dir + "/model/ResNet/KD/Student_model_" + str(int(T)) + "_" + Distance_type + "_checkpoint.tar"
        branch_model[i], best_acc = Knowledge_distillation(branch_model[i], Teacher_model, Epoch=Epoch, Distance_type=Distance_type,
                                                           Temperature=T, DataSet="cifar-100", beta=0.1, Print_epoch_fre=1)
        torch.save({
            "best_acc": best_acc,
            "state_dict": branch_model[i].state_dict(),
        }, dir)
    print (" Train KD_ResNet Over")

def Test_model(type="As"):
    print("ResNet Test ")
    # 初始化模型
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)

    train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=1)
    model_num = len(branch_model)

    for i in range(model_num):
        dir = Project_dir + "/model/ResNet/KD/ResNet_com_model_" + str(i) + "_checkpoint.tar"
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
        branch_model[i].load_state_dict(checkpoint["state_dict"])

    if torch.cuda.is_available():
        for i in range(model_num):
            branch_model[i] = branch_model[i].cuda()

    for i in range(model_num):
        pre_acc, loss, total_time, once_time = Eval_model(branch_model[i], test_loader)
        print("Eval info, Acc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
              % (pre_acc, loss, total_time, once_time * 1000))
    # for i, data in enumerate(branch_model[1].named_parameters()):
    #     name, param = data
    #     if i == 1:
    #         print (i, name, param)
    # proportion = [0.8, 0.15, 0.05]
    # exit_threshold1 = get_Exit_Threshold(branch_model, test_loader, proportion)
    # exit_threshold = [0.999, 0.999, 0.99]
    # voting_weight = [0.2, 0.5, 0.3]
    # Eval_BranchyNet(branch_model, exit_threshold, voting_weight, test_loader)
    # pre_acc, loss, total_time, once_time = Eval_model(teacher_model, test_loader)
    # print("Eval info, Acc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
    #       % (pre_acc, loss, total_time, once_time * 1000))

def Test_Teacher(T=1.0):
    print ("Test_Teacher, T = ", T)
    train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=128)
    Teacher_dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_" + str(int(T)) + "_checkpoint.tar"
    Teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)
    checkpoint = torch.load(Teacher_dir, map_location=lambda storage, loc: storage)
    Teacher_model.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.is_available():
        Teacher_model = Teacher_model.cuda()
    pre_acc, loss, total_time, once_time = Eval_model(Teacher_model, test_loader)
    print("Eval infoAcc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
          % (pre_acc, loss, total_time, once_time * 1000))


if __name__ == "__main__":

    Project_dir = my_functions.get_project_dir()
    # Train_ResNet()
    # 测试知识蒸馏的优势
    # KD_ResNet(T=1.0, Epoch=350, Distance_type="WS")
    # KD_ResNet(T=2.0, Epoch=350, Distance_type="WS")
    # KD_ResNet(T=3.0, Epoch=350, Distance_type="WS")
    KD_ResNet(T=4.0, Epoch=350, Distance_type="WS")
    # KD_ResNet(T=5.0, Epoch=350, Distance_type="WS")
    # KD_ResNet(T=6.0, Epoch=350, Distance_type="WS")
    # KD_ResNet(T=7.0, Epoch=350, Distance_type="WS")


    # KD_ResNet(T=1.0, Epoch=350, Distance_type="KL")
    # KD_ResNet(T=2.0, Epoch=350, Distance_type="KL")
    # KD_ResNet(T=3.0, Epoch=350, Distance_type="KL")
    # KD_ResNet(T=4.0, Epoch=350, Distance_type="KL")
    # KD_ResNet(T=5.0, Epoch=350, Distance_type="KL")
    # KD_ResNet(T=6.0, Epoch=350, Distance_type="KL")
    # KD_ResNet(T=7.0, Epoch=350, Distance_type="KL")
    # time.sleep(5)
    # Train_Teacher(T=1.0)
    # time.sleep(5)
    # Train_Teacher(T=4.0)
    # time.sleep(5)
    # KD_ResNet(T = 1.0, Epoch=300, Distance_type="JS")
    # time.sleep(5)
    # KD_ResNet(T=4.0, Epoch=300, Distance_type="WS")
    # Test_Teacher(T=1.0)
    # Test_Teacher(T=4.0)
