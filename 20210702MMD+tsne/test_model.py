from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import os
import math
import data_loader
import ResNet as models
from Weight import Weight
from Config import *
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

cuda = not no_cuda and torch.cuda.is_available()
#torch.manual_seed(seed)
#if cuda:
#    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
target_test_name = "DE_6205_1HP_3C_inner_inner7"
#source_loader = data_loader.load_training(root_path, source_name, 1500, kwargs)
#target_train_loader = data_loader.load_training(root_path, target_train_name, 1500, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_test_name, 300, kwargs)

#len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
#len_source_loader = len(source_loader)
len_target_loader = len(target_test_loader)
#print(len_target_dataset)
#print(len_target_loader)
num_perclass=int(len_target_dataset/class_num)
print(num_perclass)


def Full_Probability_Evaluation (output_softmax):
    Evaluation_index=np.zeros(len_target_dataset)
    output_np=output_softmax.cpu().numpy()
    #print(output_np)
    #print(output_np.shape)
    for i in range(len_target_dataset):
        #print(i)
        #print(output_np[i])
        #for j in range(class_num):
            #print(j)
        # 测试标签对应的含义为：0:Normal   1:ball14   2:ball21   3:ball7
        # 4:inner14   5:inner21   6:inner7   7:outer14   8:outer21   9:outer7
        # 计算指标时的权重为：Normal:0   ball7:4   ball14:5   ball21:6
        # inner7:7   inner14:8   inner21:9   outer7:1   outer14:2   outer21:3
        #Evaluation_index[i]=output_np[i][0]*0+output_np[i][1]*5+output_np[i][2]*6+output_np[i][3]*4+output_np[i][4]*8+output_np[i][5]*9+output_np[i][6]*7+output_np[i][7]*2+output_np[i][8]*3+output_np[i][9]*1
        #Evaluation_index[i]=output_np[i][0]*0+output_np[i][1]*2+output_np[i][2]*3+output_np[i][3]*1
        Evaluation_index[i]=output_np[i][0]*1+output_np[i][1]*3+output_np[i][2]*4
    
    #a=Evaluation_index[num_perclass*0:num_perclass*1]
    #b=Evaluation_index[num_perclass*9:num_perclass*10]
    #c=Evaluation_index[num_perclass*7:num_perclass*8]
    #d=Evaluation_index[num_perclass*8:num_perclass*9]
    #e=Evaluation_index[num_perclass*3:num_perclass*4]
    #f=Evaluation_index[num_perclass*1:num_perclass*2]
    #g=Evaluation_index[num_perclass*2:num_perclass*3]
    #h=Evaluation_index[num_perclass*6:num_perclass*7]
    #i=Evaluation_index[num_perclass*4:num_perclass*5]
    #j=Evaluation_index[num_perclass*5:num_perclass*6]
    #Evaluation_index=np.hstack((a,b,c,d,e,f,g,h,i,j))
    #Evaluation_index=np.hstack((a,e,f,g))
    print(Evaluation_index)
    return Evaluation_index
    #return output_np


if __name__ == '__main__':

    model=torch.load("Model_inner_S6205_0HP_3C_withoutIR07_T6205_1HP_final.pkl")
    #model=torch.load("Model_inner_S_6205_0HP_50Sample_T_6205_1HP_final.pkl")
    # 判断GPU是否可用
    if cuda:
        # 在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，
        # 而是需要在程序中显示指定。调用model.cuda()，可以将模型加载到GPU上去。
        model.cuda()
    
    model.eval()
    # 定义诊断准确的样本数
    correct = 0
    test_loss = 0
    with torch.no_grad():
        # 读取目标域测试样本，及其标签
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # 由于输入的第一个第二个参数相同，即将目标域数据同时作为源域数据和目标域数据输入模型
            # 由于在“源域”和“目标域”样本完全相同，故而输出的第二个参数LMMD损失：loss0=0
            # 输出的第一个值output代表对目标域测试样本预测得到的标签值
            output, loss0 = model(data, data, target)
            # 计算测试的损失值，这部分损失值完全是分类损失。
            test_loss += F.nll_loss(F.softmax(output, dim = 1), target).item() # sum up batch loss
            #test_loss += F.nll_loss(F.log_softmax(output, dim = 1), target).item() # sum up batch loss
            # 获得预测的标签。
            pred = output.data.max(1)[1] # get the index of the max log-probability
            # 计算分类争取的测试样本数目。
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            #print(pred)
            #print(pred[-1])
            #print("----------------------------------------------------------------------")
            #print("下面是没有就近取整的预测输出值")
            #print(output[0])
            output_softmax=F.softmax(output, dim = 1)
            #print(output_softmax[0])
            #for i in range(10):
                #print(output_softmax[i])
            #print(type(output_softmax.cpu().numpy()))
            #print(output_softmax.shape)
            #print(sum(output))
            HI=Full_Probability_Evaluation(output_softmax)
                
            #plt.plot(HI)
            print(HI)
            np.savetxt("HI_S_DE0HP_T_DE1HP_inner_3C_withoutIR07.txt",HI)
            
            
        # 计算得到平均损失
        #test_loss /= len_target_dataset
        #print('\n{} set: Accuracy: {}/{} ({:.6f}%)\n'.format(
        #    target_test_name, correct, len_target_dataset,
        #    100. * float(correct) / float(len_target_dataset)))


