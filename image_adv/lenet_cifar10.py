import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

torch.manual_seed(2023)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 20  # 训练整批数据的次数
BATCH_SIZE = 256
LR = 0.001  # 学习率
DOWNLOAD_CIFAR = True  # 表示还没有下载数据集，如果数据集下载好了就写False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 下载mnist手写数据集
train_data = torchvision.datasets.CIFAR10(
    root='../datasets/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
    download=DOWNLOAD_CIFAR,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.CIFAR10(
    root='../datasets/',
    train=False,  # 表明是测试集
    download=True
)

# 批训练 256个samples， 1  channel，28x28 (256,1,28,28)
# Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱数据，一般都打乱
)

# 进行测试
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor) / 255
test_x = test_x.to(device)
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.test_labels
test_y = test_y.to(device)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


lenet = LeNet().to(device)

# 优化器选择Adam
optimizer = torch.optim.Adam(lenet.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = lenet(b_x)  # 先将数据放到cnn中计算output
        loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 应用梯度

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss)

# 再正常测试样本上验证精确度
lenet.eval()
test_output = lenet(test_x)
pred_y = torch.max(test_output, 1)[1]
accuracy = (pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.4f on clean data' % accuracy)

# 使用deepfool攻击样本，验证精确度
adv_x = projected_gradient_descent(lenet, test_x, 0.1, 0.02, 20, np.Inf)
adv_output = lenet(adv_x)
adv_pred_y = torch.max(adv_output, 1)[1]
adv_accuracy = (adv_pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.4f on adv data' % adv_accuracy)

# ---------------------------------------------------------
# 下面开始进行黑盒攻击
student_EPOCH = 50
student_model = LeNet()
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)
# 先通过教师模型的接口构建学生模型的训练样本
test_output = lenet(test_x)
pred_y = torch.max(test_output, 1)[1]

for epoch in range(student_EPOCH):
    for i in range(int(test_x.shape[0] / BATCH_SIZE)):  # 分配batch data
        stu_x = test_x[i * BATCH_SIZE: (i + 1) * BATCH_SIZE].to(device)
        stu_y = pred_y[i * BATCH_SIZE: (i + 1) * BATCH_SIZE].to(device)
        output = student_model(stu_x)  # 先将数据放到cnn中计算output
        loss = loss_func(output, stu_y)  # 输出和真实标签的loss，二者位置不可颠倒
        student_optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        student_optimizer.step()  # 应用梯度

# 黑盒攻击
adv_x = projected_gradient_descent(student_model, test_x, 0.1, 0.05, 20, np.Inf)
adv_output = lenet(adv_x)
adv_pred_y = torch.max(adv_output, 1)[1]
adv_accuracy = (adv_pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.4f on adv data' % adv_accuracy)
