import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, random_split
import json
import os
from torchvision.models import resnet18, resnet50, efficientnet_b0, densenet121

batch_size_1 = 512
batch_size_2 = 1024
# 基准配置
base_config = {
    "model": "resnet18",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": batch_size_1,
    "pretrained": False,
    "augmentation": False,
    "regularization": False,
    "max_epochs": 100,  # 最大训练轮数
    "early_stop_patience": 10  # 早停容忍轮数
}

# 单变量实验设计 + 新增模型
experiments = [
    ##### 1 #####
    {**base_config, "batch_size": batch_size_1},  # 基准实验

    # 模型对比
    ##### 2 #####
    {**base_config, "model": "efficientnet_b3", "batch_size": batch_size_1},  # EfficientNet-B3
    ##### 3 #####
    {**base_config, "model": "resnet50", "batch_size": batch_size_1},  # ResNet50
    ##### 4 #####
    {**base_config, "model": "convnext_base", "batch_size": batch_size_1},  # ConvNeXt

    # 优化器对比
    ##### 5 #####
    {**base_config, "optimizer": "SGD", "batch_size": batch_size_1},  # 使用 SGD
    ##### 6 #####
    {**base_config, "optimizer": "RMSprop", "batch_size": batch_size_1},  # 使用 RMSprop
    ##### 7 #####
    {**base_config, "optimizer": "AdamW", "batch_size": batch_size_1},  # 使用 AdamW

    # 单变量调整
    ##### 8 #####
    {**base_config, "pretrained": True, "batch_size": batch_size_1},  # 启用预训练
    ##### 9 #####
    {**base_config, "augmentation": True, "batch_size": batch_size_1},  # 启用数据增强
    ##### 10 #####
    {**base_config, "regularization": True, "batch_size": batch_size_1},  # 启用正则化
    ##### 11 #####
    {**base_config, "learning_rate": 0.01, "batch_size": batch_size_1},  # 提高学习率

    # Batch Size 对比
    ##### 12 #####
    {**base_config, "batch_size": batch_size_2},

    # 组合实验
    ##### 13 #####
    {**base_config, "pretrained": True, "augmentation": True, "batch_size": batch_size_1},  # 预训练 + 数据增强
    ##### 14 #####
    {**base_config, "pretrained": True, "regularization": True, "batch_size": batch_size_1},  # 预训练 + 正则化
    ##### 15 #####
    {**base_config, "augmentation": True, "batch_size": batch_size_1, "learning_rate": 0.01},  # 数据增强 + 高学习率
    ##### 16 #####
    {**base_config, "model": "efficientnet_b3", "optimizer": "AdamW", "pretrained": True, "batch_size": batch_size_1},  # EfficientNet-B3 + AdamW + 预训练
    ##### 17 #####
    {**base_config, "model": "convnext_base", "optimizer": "AdamW", "pretrained": True, "batch_size": batch_size_1},  # ConvNeXt + SGD + 高学习率


    # Stronger model with appropriate settings
    {**base_config, "model": "efficientnet_b3", "optimizer": "AdamW", "pretrained": True, "augmentation": True, "regularization": True},
    {**base_config, "model": "convnext_base", "optimizer": "AdamW", "pretrained": True, "augmentation": True, "regularization": True},

    # Experiment with learning rate and optimizer
    {**base_config, "optimizer": "SGD", "learning_rate": 0.005, "augmentation": True, "pretrained": True},
    {**base_config, "optimizer": "AdamW", "learning_rate": 0.0005, "augmentation": True, "pretrained": True},

    # Introduce advanced augmentation techniques
    {**base_config, "model": "efficientnet_b3", "augmentation": True, "pretrained": True, "regularization": False, "learning_rate": 0.001},
    {**base_config, "model": "convnext_base", "augmentation": True, "pretrained": True, "regularization": False, "learning_rate": 0.001},

    # Batch size experiments
    {**base_config, "model": "efficientnet_b3", "batch_size": 256, "optimizer": "AdamW", "pretrained": True, "augmentation": True},
    {**base_config, "model": "convnext_base", "batch_size": 256, "optimizer": "AdamW", "pretrained": True, "augmentation": True}
]


# 日志文件路径
log_file_path = "experiment_results_with_train_accuracy.json"
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as f:
        json.dump({"experiments": []}, f, indent=4)

# 数据增强函数
def get_transforms(augmentation):
    if augmentation:
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# 加载模型
def get_model(model_name, pretrained, num_classes, regularization):
    if model_name == "vgg16":
        model = torchvision.models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(batch_size_1, num_classes)
    elif model_name == "resnet50":
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_name == "efficientnet_b3":
        model = torchvision.models.efficientnet_b3(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1536, num_classes)
    elif model_name == "densenet121":
        model = densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(1024, num_classes)
    elif model_name == "convnext_base":
        model = torchvision.models.convnext_base(pretrained=pretrained)
        model.classifier[2] = nn.Linear(1024, num_classes)
    # elif model_name == "vit_base_patch16_224":
    #     model = torchvision.models.vit_base_patch16_224(pretrained=pretrained)
    #     model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # 添加正则化（如果启用）
    if regularization:
        if model_name.startswith("resnet") or model_name.startswith("efficientnet"):
            feature_dim = batch_size_1 if model_name == "resnet18" else 1280
            if model_name == "efficientnet_b3":
                feature_dim = 1536
            model.fc = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif model_name == "convnext_base":
            model.classifier[2] = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif model_name == "vit_base_patch16_224":
            model.heads.head = nn.Sequential(
                nn.Linear(model.heads.head.in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

    return model

# 训练函数
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for imgs, targets in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return running_loss / len(dataloader), correct / total

# 验证函数
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0

# 实验执行
for idx, exp in enumerate(experiments):
    exp["experiment_id"] = idx + 1
    print(f"Running Experiment ID: {exp['experiment_id']} with config: {exp}")
    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root="data", train=True, transform=get_transforms(exp["augmentation"]), download=True
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root="data", train=False, transform=get_transforms(False), download=True
        )
        train_loader = DataLoader(train_dataset, batch_size=exp["batch_size"], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=exp["batch_size"], shuffle=False, num_workers=2)

        # 加载模型
        model = get_model(exp["model"], exp["pretrained"], num_classes=10, regularization=exp["regularization"])
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=exp["learning_rate"]) if exp[
                                                                                         "optimizer"] == "Adam" else torch.optim.SGD(
            model.parameters(), lr=exp["learning_rate"], momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(patience=exp["early_stop_patience"], delta=0.001)
        results = {"train_loss": [], "val_accuracy": [], "train_accuracy": []}

        for epoch in range(exp["max_epochs"]):
            train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                                         "cuda" if torch.cuda.is_available() else "cpu")
            val_accuracy = evaluate(model, val_loader, "cuda" if torch.cuda.is_available() else "cpu")
            results["train_loss"].append(train_loss)
            results["train_accuracy"].append(train_accuracy)
            results["val_accuracy"].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{exp['max_epochs']}: Train Loss = {train_loss:.4f}, "
                  f"Train Accuracy = {train_accuracy:.4f}, Val Accuracy = {val_accuracy:.4f}")

            early_stopping(val_accuracy)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # 保存结果并打印
        experiment_result = {**exp, "results": results}
        with open(log_file_path, "r") as f:
            log_data = json.load(f)
        log_data["experiments"].append(experiment_result)
        with open(log_file_path, "w") as f:
            json.dump(log_data, f, indent=4)

        print(f"Experiment ID {exp['experiment_id']} results saved to {log_file_path}")
        print(json.dumps(experiment_result, indent=4))

    except Exception as e:
        print(f"Error occurred in Experiment ID {exp['experiment_id']}: {e}")