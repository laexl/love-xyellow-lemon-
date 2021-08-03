from torchvision import datasets, transforms
import torch

def load_training(root_path, dir, batch_size, kwargs):

    # Compose的功能是将多个transforms组合起来。
    transform = transforms.Compose(
        # 将输入的图片大小重新调整。
        [transforms.Resize([256, 256]),
        # 在随机位置裁剪给定的图像，将图像裁剪成224*224的大小。
         transforms.RandomCrop(224),
         # 以给定的概率水平反转输入的图像。默认概率为0.5。
         transforms.RandomHorizontalFlip(),
         # 将图像转换成tensor类型，范围从0-225转化为0-1
         transforms.ToTensor()])
    # 通用数据加载器，第一个参数是数据来源，第二个数据是对图片进行一定处理变换。
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    # DataLoader：组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
    # 第一个参数dataset是数据来源，第二个参数batch_size是每一批次的图片样本数量大小
    # 第三个参数shuffle表示是否打乱数据顺序。
    # 第四个参数drop_last，如果数据集大小不能整除批次，剩下的余数如何处理。
    # 如果是True则删除最后一个不完整批次，如果是false则最后一个批次小一点。
    # 最后一个参数kwargs表示其他参数，在DSAN中设置为：
    # num_workers': 1, 'pin_memory': True
    # num_workers表示要用于数据加载的子进程数，0表示数据将加载到主进程中。
    # pin_memory如果为真，则数据加载程序将在返回张量之前将其复制到CUDA固定内存中
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

# 加载测试用的数据集
def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=root_path + dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader