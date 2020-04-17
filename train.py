import torch.nn as nn
import torch
import os
import sys
import argparse
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import torch.cuda


CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CHECK_POINT_PATH = "./checkpoint"
MILESTONES = [60, 120, 160]


def training():
    net.train()
    length = len(trainloader)
    total_sample = len(trainloader.dataset)
    total_loss = 0
    correct = 0
    for step, (x, y) in enumerate(trainloader):
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()

        output = net(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predict = torch.max(output, 1)
        correct += (predict == y).sum()

        fstep.write("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}\n".format(
                epoch+1, step+1, step*args.b + len(y), total_sample, loss.item()
            ))
        fstep.flush()

        if step % 10 == 0:
            print("Epoch:{}\t Step:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}".format(
                epoch+1, step+1, step*args.b + len(y), total_sample, loss.item()
            ))

    fepoch.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
        epoch + 1, total_loss/length, optimizer.param_groups[0]['lr'], float(correct)/ total_sample
    ))
    fepoch.flush()


def evaluating():
    net.eval()
    length = len(testloader)
    total_sample = len(testloader.dataset)
    total_loss = 0
    correct = 0
    inference_time = 0
    for step, (x, y) in enumerate(testloader):
        x = x.cuda()
        y = y.cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = net(x)
        _, predict = torch.max(output, 1)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        inference_time += start.elapsed_time(end)  # milliseconds

        loss = loss_function(output, y)

        total_loss += loss.item()

        correct += (predict == y).sum()

    acc = float(correct) / total_sample
    feval.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
        epoch + 1, total_loss / length, optimizer.param_groups[0]['lr'], acc
    ))
    feval.flush()
    return acc, total_loss/length, inference_time


if __name__ == '__main__':
    # arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", default='mobilenet', help='net type')
    parser.add_argument("-b", default=128, type=int, help='batch size')
    parser.add_argument("-lr", default=0.1, help='initial learning rate', type=int)
    parser.add_argument("-e", default=200, help='EPOCH', type=int)
    parser.add_argument("-optim", default="SGD", help='optimizer')
    args = parser.parse_args()

    # data processing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
    ])

    traindata = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
    testdata = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

    trainloader = DataLoader(traindata, batch_size=args.b, shuffle=True, num_workers=2)
    testloader = DataLoader(testdata, batch_size=args.b, shuffle=True, num_workers=2)

    # define net
    if args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(1, 100).cuda()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(1, 100).cuda()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet([4, 8, 4], 3, 1, 100).cuda()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(100, 1).cuda()
    elif args.net == 'efficientnetb0':
        from models.efficientnet import efficientnet
        print("loading net")
        net = efficientnet(1, 1, 100, bn_momentum=0.9).cuda()
        print("loading finish")
    else:
        print('We don\'t support this net.')
        sys.exit()

    # define loss, optimizer, lr_scheduler and checkpoint path
    print("defining training")
    loss_function = nn.CrossEntropyLoss()
    if args.optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2, last_epoch=-1)
    time = str(datetime.date.today() + datetime.timedelta(days=1))
    checkpoint_path = os.path.join(CHECK_POINT_PATH, args.net, time)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print("defining finish")
    # train and eval
    best_acc = 0
    total_time = 0
    with open(os.path.join(checkpoint_path, 'EpochLog.txt'), 'w') as fepoch:
        with open(os.path.join(checkpoint_path, 'StepLog.txt'), 'w') as fstep:
            with open(os.path.join(checkpoint_path, 'EvalLog.txt'), 'w') as feval:
                with open(os.path.join(checkpoint_path, 'Best.txt'), 'w') as fbest:
                    print("start training")
                    for epoch in range(args.e):
                        training()
                        print("evaluating")
                        accuracy, averageloss, inference_time = evaluating()

                        scheduler.step()

                        print("saving regular")
                        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'regularParam.pth'))

                        if accuracy > best_acc:
                            print("saving best")
                            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'bestParam.pth'))

                            fbest.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
                                epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy
                            ))
                            fbest.flush()
                            best_acc = accuracy
                        # print(inference_time)
                        total_time += (inference_time/len(testloader.dataset))
    print(total_time)
    print(total_time / args.e)











