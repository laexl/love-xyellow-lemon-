from __future__ import print_function
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
import data_loader
import resnest as models
from Weight import Weight
from Config import *
import time
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
import numpy as np
import matplotlib
matplotlib.use('TkAgg', warn=False, force=True)
import matplotlib.pyplot as plt

all_loss=np.zeros(epochs)

cuda = not no_cuda and torch.cuda.is_available()
#torch.manual_seed(seed)
#if cuda:
#    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

source_loader = data_loader.load_training(root_path, source_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_train_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_test_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)

# 定义模型训练函数，传入两个参数，第一个是当前的迭代数，第二个是待训练模型
def train(epoch, model):
    # 定义当前迭代的学习率LEARNING_RATE，lr为初始学习率，在Config.py中定义，
    # 随着迭代次数的增加，LEARNING_RATE的值不断降低。
    LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
    # 输出当次迭代的学习率。
    print('learning rate{: .4f}'.format(LEARNING_RATE) )
    
    # 此处加上if-else条件语句的原因是为了传递参数。
    # 根据bottle_neck数值的不同，在resnet50后面添加新的层。bottle_neck在config.py中定义。
    # 当bottle_neck为true时，以两个线性变换函数Linear为两个全连接层，
    # 第一个全连接层输入2048，输出256；第二个全连接层输入256，输出的数量由分类数传入。
    if bottle_neck:
        # 实现随机梯度下降，可选地使用动量(动量可以加快学习过程，对于高曲率、
        # 小但一致的梯度或者噪声较大的梯度可以很好的加快学习过程)
        optimizer = torch.optim.SGD([
            # 参数可优化或决定参数组的定义
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},], 
            # 把当前学习率赋值给lr，动量momentum在config.py中定义。
            # weight_decay权重衰减系数，也在config.py中定义。
            lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)
    else:
    # 当bottle_neck为false时，只连接一个全连接层。
    # 全连接层输入2048，输出的数量由分类数传入。
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': LEARNING_RATE},
            ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    # 告诉网络，开始训练模式。
    model.train()

    # 将data_loader类型的源域训练样本source_loader转化为iter可迭代对象。
    iter_source = iter(source_loader)
    # 将data_loader类型的目标域训练样本target_train_loader转化为iter可迭代对象。
    iter_target = iter(target_train_loader)
    # 加载批次数
    num_iter = len_source_loader
    # 逐批开始训练
    for i in range(1, num_iter):
        # 加载一批次源域数据的数据和标签。
        data_source, label_source = iter_source.next()
        # 加载一批次目标域数据的数据，为了凸显无监督训练，不加载目标域标签。
        data_target, _ = iter_target.next()
        
        # 这个判断结构的用处不明。
        # 如果当前源域数据的批次数是目标域总批次数的整数倍
        if i % len_target_loader == 0:
            iter_target = iter(target_train_loader)
        
        #如果Cuda可用将源域样本、源域标签、目标域样本都转化成cuda版本。
        if cuda:
            #print('此时正在使用Cuda')
            data_source, label_source = data_source.cuda(), label_source.cuda()
            data_target = data_target.cuda()
        # 转化成变量
        data_source, label_source = Variable(data_source), Variable(label_source)
        data_target = Variable(data_target)
        # 手动梯度置零
        optimizer.zero_grad()
        # 利用ResNet.py中DSAN类，计算源域样本的预测标签和LMMD损失，并返回。
        label_source_pred, loss_mmd = model(data_source, data_target, label_source)
        # 将全连接层输出结果输入softmax进行归一化，
        # 根据分类器结果和源域标签对比，得到分类损失（负对数自然损失）。
        loss_cls = F.nll_loss(F.log_softmax(label_source_pred, dim=1), label_source)
        # lambd 是系数，随epoch增大，lambda也增大。
        # 即前期更注重分类损失，后期更注重MMD损失。
        #lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
        # 前期更注重MMD，随epoch增大，lambd降低，后期更注重分类损失。
        lambd = 2 - 2 / (1 + math.exp(-10 * (epoch) / epochs))
        # 将分类损失与LMMD损失相结合，定义目标函数
        loss = loss_cls + param * lambd * loss_mmd
        # 以下两句经常组合使用，前者计算梯度，后者利用梯度更新网络参数。
        # 反向传播，计算当前梯度
        loss.backward()
        # 使用当前参数空间对应的梯度对网络参数进行更新。
        optimizer.step()
        # log_interval在Config.py中定义，
        # 每隔log_interval个epoch就输出当前epoch、训练进度、各项损失值。
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}'.format(
                epoch, i * len(data_source), len_source_dataset,
                100. * i / len_source_loader, loss.item(), loss_cls.item(), loss_mmd.item()))

def test(model):
    # 功能与model.train类似，但是不启用批标准化BatchNormalization和dropout。
    # 一旦模型由数据输入，即使不训练，也会改变权值，这是BN层带来的性质。
    model.eval()
    # 定义测试的损失值
    test_loss = 0
    # 定义诊断准确的样本数
    correct = 0
    # torch.no_grad()可以作为装饰器，在网络测试的函数前加上。
    # 对于tensor类型的数据，每次进行运算，都会被记录(如加减乘除)
    # 在with torch.no_grad()包裹下的语句，不会记录运算类型。因此也不会计算梯度？
    # a = torch.tensor([1.1], requires_grad=True)
    # b = a * 2
    # Out[63]: tensor([2.2000], grad_fn=<MulBackward0>)
    # b.add_(2)
    # Out[64]: tensor([4.2000], grad_fn=<AddBackward0>)
    # with torch.no_grad():
    #     b.mul_(2)
    # Out[66]: tensor([8.4000], grad_fn=<AddBackward0>)
    with torch.no_grad():
        # 读取目标域测试样本，及其标签
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # 由于输入的第一个第二个参数相同，即将目标域数据同时作为源域数据和目标域数据输入模型
            # 由于在“源域”和“目标域”样本完全相同，故而输出的第二个参数LMMD损失：t_output=0
            # 输出的第一个值s_output代表对目标域测试样本预测得到的标签值
            s_output, t_output = model(data, data, target)
            # 计算测试的损失值，这部分损失值完全是分类损失。
            test_loss += F.nll_loss(F.log_softmax(s_output, dim = 1), target).item() # sum up batch loss
            # 获得预测的标签。
            pred = s_output.data.max(1)[1] # get the index of the max log-probability
            # 计算分类争取的测试样本数目。
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        # 计算得到平均损失
        test_loss /= len_target_dataset
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            target_test_name, test_loss, correct, len_target_dataset,
            100. * float(correct) / float(len_target_dataset)))
    return correct,test_loss

# 主函数
if __name__ == '__main__':
    # 实例化一个DSAN类(即一个DSAN模型，传递的参数是类别数，在Config.py中定义)
    model = models.DSAN(num_classes=class_num)
    # 定义变量correct，用来存储历史最佳的故障诊断准确率。
    correct = 0
    # 输出模型结构
    print(model)
    # 判断GPU是否可用
    if cuda:
        # 在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，
        # 而是需要在程序中显示指定。调用model.cuda()，可以将模型加载到GPU上去。
        model.cuda()
    # 开始计时
    time_start=time.time()
    # 开始迭代，epochs是迭代次数，epoch是当前迭代次数。
    # epochs在Config.py中被定义。
    for epoch in range(1, epochs + 1):
        # 调用训练模型的程序train()。传入的参数是当前的迭代次数和待训练的模型。
        train(epoch, model)
        # 定义当前正确分类样本数t_correct，调用测试函数，把测试的正确样本数存储进t_correct
        t_correct,test_loss = test(model)
        all_loss[epoch-1]=test_loss
        # 判断当次测试分类正确样本数t_correct是否高于历史最佳分类正确样本数，
        # 如果是，将该分类正确样本数值存入最佳分类正确样本数correct
        if t_correct > correct:
            correct = t_correct
            # 存储准确率最佳时的模型。
            #torch.save(model, 'Model_S6205_1HP_T6205_2HP_100_best.pkl')
        # 当前迭代结束，结束计时。
        end_time = time.time()
        # 输出源域、目标域、最大准确率以及准确分类样本数和总测试样本数
        print('source: {} ;target_train: {};target_test:{} max correct: {} max accuracy{: .6f}%\n'.format(
              source_name, target_train_name, target_test_name, correct, 100. * correct / len_target_dataset ))
        # 结束计时时间-开始计时时间=模型从开始运行到当次迭代总计用时。
        print('cost time:', end_time - time_start)
    #torch.save(model, 'Model_S6205_1HP_T6205_0HP_100_final.pkl')
    #print(all_loss)
    #mytext = open('S6205_0HP_T6205_1HP_loss.txt','w')
    #mytext.write(all_loss.astype(str))
    #plt.figure(figsize=(1500,1000))     
    #plt.plot(all_loss)
    #plt.ylabel('Average loss of Epoch')
    #plt.xlabel('Number of Epoch')    
    #plt.show()
    #plt.savefig('epoch_loss_S6205_0HP_T6205_1HP.png')