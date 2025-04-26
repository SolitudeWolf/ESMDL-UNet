import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义数据集路径
data_dir = 'path_to_your_train_dataset'

# 定义数据预处理（不包括Normalize）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 根据你的模型输入尺寸调整
    transforms.ToTensor()
])

# 加载数据集
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# 计算均值和标准差
mean = 0.0
std = 0.0
nb_samples = 0.0

for data, _ in dataloader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f'Mean: {mean}')
print(f'Std: {std}')
