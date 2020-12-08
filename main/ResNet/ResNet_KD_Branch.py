import sys
sys.path.append("../..")
sys.path.append("..")
import torch
from functions import my_functions
from model.ResNet import get_ResNet_model
from functions.my_functions import Train_model, Eval_model
from functions.my_functions import Knowledge_distillation
from functions.branch_functions import get_Exit_Threshold, Eval_BranchyNet, Train_BranchyNet_Synchronization, Train_BranchyNet_Asynchronous, \
    Train_BranchyNet_Asynchronous_Back, Train_BranchyNet_Asynchronous_KD, Train_BranchyNet_Asynchronous_KD_Back
from datasets import get_cifar_100

def Train_As():

    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    print("Train_BranchyNet_Asynchronous Begin")

    # dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_As_model_0_checkpoint.tar"
    # checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    # branch_model[0].load_state_dict(checkpoint["state_dict"])

    # 直接训练其他两个网络
    branch_model = Train_BranchyNet_Asynchronous_Back(branch_model, Epoch_list=[300, 250, 250],
                                                      copy_range=[-1, 38, 44], DataSet="cifar-100", Print_epoch_fre=1, save_dir="ResNet")
    model_num = len(branch_model)
    # 保存模型
    for i in range(model_num):
        dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_As_model_" + str(i)+ "_checkpoint.tar"
        torch.save({
            "state_dict": branch_model[i].state_dict(),
        }, dir)

def Train_Sy():

    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    print("Train_BranchyNet_Asynchronous Begin")
    branch_model = Train_BranchyNet_Synchronization(branch_model, Epoch=250, copy_range=[23, 38, 0], DataSet="cifar-100", Print_epoch_fre=1, save_dir="ResNet")
    model_num = len(branch_model)
    # 保存模型
    for i in range(model_num):
        dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_Sy_model_" + str(i)+ "_checkpoint.tar"
        torch.save({
            "state_dict": branch_model[i].state_dict(),
        }, dir)

def Train_As_main_model():

    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    print("Train_BranchyNet_Asynchronous Begin")
    dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_As_model_2_checkpoint.tar"
    checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    branch_model[2].load_state_dict(checkpoint["state_dict"])
    # 直接训练其他两个网络
    branch_model = Train_BranchyNet_Asynchronous(branch_model, main_Epoch=200, branch_Epoch=150, copy_range=[23, 38, 0], DataSet="cifar-100", Print_epoch_fre=1, save_dir="VGGNet")
    model_num = len(branch_model)
    # 保存模型
    for i in range(model_num-1):
        dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_As_model_" + str(i)+ "_checkpoint.tar"
        torch.save({
            "state_dict": branch_model[i].state_dict(),
        }, dir)

def KD_ResNet(Beta=0.1, T = 1.0, Distance_type="WS", type="As"):
    # 获得教师和学生模型
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)
    print("Train_BranchyNet_Asynchronous Begin")
    dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_" + str(int(T)) + "_checkpoint.tar"
    if torch.cuda.is_available():
        teacher_model = teacher_model.cuda()
        checkpoint = torch.load(dir)
    else:
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    teacher_model.load_state_dict(checkpoint["state_dict"])

    # dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_KD_As_model_2_checkpoint.tar"
    # if torch.cuda.is_available():
    #     branch_model[2] = branch_model[2].cuda()
    #     checkpoint = torch.load(dir)
    # else:
    #     checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    # branch_model[2].load_state_dict(checkpoint["state_dict"])

    # 直接训练分支网络
    # branch_model = Train_BranchyNet_Asynchronous_KD(branch_model, teacher_model, main_Epoch=300, branch_Epoch=150, copy_range=[23, 38, -1], DataSet="cifar-100",
    #                                  Print_epoch_fre = 1, Distance_type=Distance_type, Temperature=4.0, beta=Beta)
    branch_model = Train_BranchyNet_Asynchronous_KD(branch_model, teacher_model, Epoch_list=[300, 250, 250], copy_range=[38, 44, -1],
                                                    DataSet="cifar-100", Print_epoch_fre=1, Distance_type=Distance_type, Temperature=4.0, beta=Beta)
    model_num = len(branch_model)
    # 保存模型
    for i in range(model_num):
        # dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_KD_As_WS_model_" + str(i) + "_checkpoint.tar"
        dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_KD_"+ type + "_" + Distance_type + "_model_" + str(i) + "_checkpoint.tar"
        torch.save({
            "state_dict": branch_model[i].state_dict(),
        }, dir)
    print ("\n")

def KD_ResNet_back(Beta=0.1, T = 1.0, Distance_type="WS", type="As"):
    # 获得教师和学生模型
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)
    print("Train_BranchyNet_Asynchronous Back Begin")
    # dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_4_checkpoint.tar"
    dir = Project_dir + "/model/ResNet/KD/ResNet_Teacher_model_" + str(int(T)) + "_checkpoint.tar"
    if torch.cuda.is_available():
        teacher_model = teacher_model.cuda()
        checkpoint = torch.load(dir)
    else:
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
    teacher_model.load_state_dict(checkpoint["state_dict"])

    branch_model = Train_BranchyNet_Asynchronous_KD_Back(branch_model, teacher_model, Epoch_list = [250, 250, 250], copy_range=[-1, 38, 44],
                                                         DataSet="cifar-100",Print_epoch_fre=1, Distance_type=Distance_type, Temperature=4.0,beta=Beta)
    model_num = len(branch_model)
    # 保存模型
    for i in range(model_num):
        # dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_KD_As_KD_Back_model_" + str(i) + "_back_checkpoint.tar"
        dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_KD_" + type + "_" + Distance_type + "_model_" + str(i) + "_back_checkpoint.tar"
        torch.save({
            "state_dict": branch_model[i].state_dict(),
        }, dir)
    print ("\n")

def Test_model(type="As"):
    print("ResNet Test ")
    # 初始化模型
    main_model, branch_model = get_ResNet_model.get_model(num_classes=100)
    teacher_model = get_ResNet_model.get_teacher_model(num_classes=100)

    train_loader, test_loader = get_cifar_100.get_data(train_batch_size=128, test_batch_size=1)
    model_num = len(branch_model)

    # teacher_dir = Project_dir + "/model/VGGNet/KD/VGGNet_Teacher_model_1_checkpoint.tar"
    # teacher_checkpoint = torch.load(teacher_dir, map_location=lambda storage, loc: storage)
    # teacher_model.load_state_dict(teacher_checkpoint["state_dict"])

    for i in range(model_num):
        if type=="As":
            dir = Project_dir + "/model/ResNet/KD/ResNet_As_model_" + str(i) + "_checkpoint.tar"
        else:
            dir = Project_dir + "/model/ResNet/KD_Branch/ResNet_As_Back/ResNet_Sy_model_" + str(i) + "_checkpoint.tar"
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
        # checkpoint = torch.load(dir)
        branch_model[i].load_state_dict(checkpoint["state_dict"])
        if torch.cuda.is_available():
            branch_model[i] = branch_model[i].cuda()
    # for i, data in enumerate(branch_model[1].named_parameters()):
    #     name, param = data
    #     if i == 1:
    #         print (i, name, param)
    proportion = [0.9, 0.05, 0.05]
    exit_threshold = get_Exit_Threshold(branch_model, test_loader, proportion)
    print (exit_threshold)
    # exit_threshold = [0.95,0.97,0.98]
    # voting_weight = [0.30, 0.30, 0.40]
    # Eval_BranchyNet(branch_model, exit_threshold, voting_weight, test_loader)
    # pre_acc, loss, total_time, once_time = Eval_model(teacher_model, test_loader)
    # print("Eval info, Acc: %3.2f, Loss:%3.5f, used_time:%.2fs, one_used_time:%.4fms"
    #       % (pre_acc, loss, total_time, once_time * 1000))




if __name__ == "__main__":

    Project_dir = my_functions.get_project_dir()
    # Train_As()
    # Train_As_main_model()
    # KD_ResNet()
    # time.sleep(5)
    # Train_Sy()
    # time.sleep(5)
    Test_model()
    # KD_ResNet_back()
