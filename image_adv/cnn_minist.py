import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


torch.manual_seed(2023)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 5  # 训练整批数据的次数
BATCH_SIZE = 256
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False  # 表示还没有下载数据集，如果数据集下载好了就写False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 下载mnist手写数据集
train_data = torchvision.datasets.MNIST(
    root='../datasets/',  # 保存或提取的位置  会放在当前文件夹中
    train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
    transform=torchvision.transforms.ToTensor(),  # 转换PIL.Image or numpy.ndarray
    download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
)

test_data = torchvision.datasets.MNIST(
    root='../datasets/',
    train=False  # 表明是测试集
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


# model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution layer 1 (（w - f + 2 * p）/ s ) + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolution layer 2
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.batch3 = nn.BatchNorm2d(8)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully-Connected layer 1

        self.fc1 = nn.Linear(128, 64)
        self.fc1_relu = nn.ReLU()

        # Fully-Connected layer 2
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # conv layer 1 的前向计算，3行代码
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)

        out = self.maxpool1(out)

        # conv layer 2 的前向计算，4行代码
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batch3(out)

        out = self.maxpool2(out)

        # Flatten拉平操作
        out = out.view(out.size(0), -1)

        # FC layer的前向计算（2行代码）
        out = self.fc1(out)
        out = self.fc1_relu(out)

        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


cnn = CNNModel().to(device)

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted

# 开始训练
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
        b_x = b_x.to(device)
        b_y = b_y.to(device)
        output = cnn(b_x)  # 先将数据放到cnn中计算output
        loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
        optimizer.zero_grad()  # 清除之前学到的梯度的参数
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 应用梯度

        if step % 100==0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss)

# 再正常测试样本上验证精确度
cnn.eval()
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1]
accuracy = (pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.4f on clean data' % accuracy)

# 使用deepfool攻击样本，验证精确度
adv_x = projected_gradient_descent(cnn, test_x, 0.1, 0.05, 20, np.Inf)
adv_output = cnn(adv_x)
adv_pred_y = torch.max(adv_output, 1)[1]
adv_accuracy = (adv_pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.4f on adv data' % adv_accuracy)


# ---------------------------------------------------------
# 下面开始进行黑盒攻击
student_EPOCH = 50
student_model = CNNModel()
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=LR)
# 先通过教师模型的接口构建学生模型的训练样本
test_output = cnn(test_x)
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
adv_x = projected_gradient_descent(student_model, test_x, 0.1, 0.1, 20, np.inf)
adv_output = cnn(adv_x)
adv_pred_y = torch.max(adv_output, 1)[1]
adv_accuracy = (adv_pred_y.data == test_y.data).sum() / test_y.shape[0]
print('test accuracy: %.4f on adv data' % adv_accuracy)
