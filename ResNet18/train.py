import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tqdm import tqdm
from pathlib import Path
import argparse

from architecture import ResNet18

def saveImages(imgFolder, classes, save_dir, num_imgs):
    for i in range(num_imgs):
        imag, clsid = imgFolder.__getitem__(i)
        pltimg = imag.permute(1,2,0).numpy()
        img = (pltimg*255.0).astype(np.uint8)
        cv2.imwrite(f'{save_dir}/{classes[clsid]}_{i}.jpg', img)

def savePlot(train_losses, test_losses, train_acc_list, test_acc_list, save_dir):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Testing Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(test_acc_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig(f"{save_dir}/Loss&Accuracy.png", dpi=300, bbox_inches="tight")
    print(f'[INFO] Training loss and accuracy curve is saved at {save_dir}')

def trainEngine(model, num_epochs, trainloader, testloader, criterion, optimizer, scheduler,device, save_dir):
    train_losses, test_losses, train_acc_list, test_acc_list = [], [], [], []
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_test = 0.0
        correct, total = 0, 0
        pbar = tqdm(trainloader)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * inputs.size(0)
            running_loss += batch_loss
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            info = f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {batch_loss:.3f}'
            pbar.set_description(info)
        
        train_loss = running_loss / len(trainloader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        model.eval()
        correct, total = 0, 0
        test_pbar = tqdm(testloader, desc="Evaluation")
        with torch.no_grad():
            for inputs, labels in test_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                test_loss = criterion(outputs, labels)
                batch_loss_test = test_loss.item()*inputs.size(0)
                running_loss_test+=batch_loss_test

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_ls = running_loss_test/len(testloader.dataset)
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)
        test_losses.append(test_ls)

        print(f'Test Accuracy : {test_acc:.3f}%')
        scheduler.step()
        
        torch.save(model.state_dict(), f'{save_dir}/epoch_{epoch+1}_acc_{test_acc:.2f}%_resnet18_CIFAR10.pt')

        savePlot(train_losses, test_losses,  train_acc_list, test_acc_list, save_dir)

    torch.save(model.state_dict(), f'{save_dir}/resnet18_CIFAR10.pt')
    print(f'[INFO] Trained weights are save at {save_dir}')

    return train_losses, test_losses, train_acc_list, test_acc_list

def main():
    parser = argparse.ArgumentParser(description='Arguments to train the resnet model.')

    parser.add_argument('--device', type=str, default='cuda', help='whether to use cuda or cpu')
    parser.add_argument('--epochs', type=int, default=60, help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='total training epochs')
    parser.add_argument('--num_imgs', type=int, default=50, help='number of testing images which will be saved separately so that it can be used during inference')
    parser.add_argument('--dir', type=str, default='store', help='directory to save weights, test images and log picture')
    
    args = parser.parse_args()

    savedir = Path(args.dir)
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)

    save_wts_dir = savedir/"weights"
    save_test_images = savedir/"cifar10_images"

    save_wts_dir.mkdir(parents=True, exist_ok=True)
    save_test_images.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    print(f'Device Used : {device}')

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    classes = trainset.classes
    print(f'[INFO] Classes in cifar10 dataset : {classes}')

    saveImages(testset, classes, save_test_images, args.num_imgs)
    print(f'[INFO] {args.num_imgs} images are saved at {save_test_images} directory for the inference.')

    model = ResNet18(num_classes=len(classes))

    trainable,non_trainable = 0, 0
    for para in model.parameters():
        trainable+=para.numel() if para.requires_grad else 0
        non_trainable+=para.numel if not para.requires_grad else 0
    print(f'[INFO] Trainable Parameters : {trainable/1e06} M.')
    print(f'[INFO] Non-Trainable Parameters : {non_trainable/1e06} M.')
    print(f'[INFO] Total Parameters : {(trainable+non_trainable)/1e06} M.')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    trainEngine(model, args.epochs, trainloader, testloader, criterion, optimizer, scheduler, device, save_wts_dir)

if __name__ == "__main__":
    main()