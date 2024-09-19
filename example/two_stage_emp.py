import os
import json
import shutil
from utils.logger import Logger
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from datetime import date
import model.resnet_cbam as resnet_cbam
from ConLoss import SupConLoss
import torch.nn as nn
import sys
from set_logger import set_logger1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(2023)  # 为所有GPU设置随机种子
    log = set_logger1(log_path='./process.log')

    epochs = 3000
    contrastive_epochs = 1000  # 训练对比损失的轮数
    classifier_epochs = 1000  # 训练分类器的轮数

    save_path = "./runs/CICM+MSL(new two stages)-结果"
    batch_size = 16

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log.info("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.36049452, 0.36049452, 0.36049452], [0.044771854, 0.044771854, 0.044771854])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.36049452, 0.36049452, 0.36049452], [0.044771854, 0.044771854, 0.044771854])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "datasets/训练数据集"))
    image_path = os.path.join(data_root, "", "")  # flower data set path 1.flower_data 2.iron
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # 加载数据集
    train_datasets = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform['train'])
    val_datasets = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform['val'])
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=8)

    iron_list = train_datasets.class_to_idx
    log.info(iron_list)
    cla_dict = dict((val, key) for key, val in iron_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open(save_path + '/' + 'class_indices-iron.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    log.info('Using {} dataloader workers every process'.format(nw))

    net = resnet_cbam.resnet_new_CICM_MSL(num_classes=11).to(device)
    model_weight_path = "resNet50.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    ckpt = torch.load(model_weight_path, map_location='cpu')
    ckpt.pop("fc.weight")
    ckpt.pop("fc.bias")
    net.load_state_dict(ckpt, strict=False)

    Sup_loss = SupConLoss(temperature=0.07)
    loss_function = nn.CrossEntropyLoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    for epoch in range(contrastive_epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloaders, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, features = net(images.to(device))
            loss = 0.8*Sup_loss(features) + 0.2*loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, contrastive_epochs, loss)
        log.info('[epoch %d] train_loss: %.3f' % (epoch + 1, running_loss / len(train_dataloaders)))

    # torch.save(net.state_dict(), save_path + '/' + "CICM+MSL(new two stages)_train_best.pth")

    for param in net.parameters():
        param.requires_grad = False

    net.fc.weight.requires_grad = True
    net.fc.bias.requires_grad = True

    optimizer = optim.Adam([net.fc.weight, net.fc.bias], lr=0.0001)
    best_acc = 0.0
    for epoch in range(classifier_epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloaders, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits, features = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, classifier_epochs, loss)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloaders, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs, features = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, classifier_epochs)

        val_accurate = acc / len(val_dataloaders)
        log.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (
        epoch + 1, running_loss / len(train_dataloaders), val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path + '/' + "CICM+MSL2(new two stages)_classifier_best.pth")

    log.info('Finished Training')


if __name__ == '__main__':
    main()
