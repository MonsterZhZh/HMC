import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter

import argparse
import copy
import time

from dataset import CubDataset, HRSCDataset
from transforms import makeDefaultTransforms


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--batch', default=32, type=int,
                        help='batch size (default: 32)')
    parser.add_argument('--epoch', default=60, type=int,
                        help='Epochs (default: 60)')
    parser.add_argument('--worker', default=4, type=int,
                        help='number of workers (default: 4)')
    parser.add_argument('--device', type=str, default='0',
                        help='Specify CUDA Device')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth',
                        help='Path of pre-trained model')
    parser.add_argument('--proportion', type=float, default=1.0,
                        help='Proportion of relabel')
    parser.add_argument('--lr', type=float, default=0.008,
                        help='learning rate')
    args = parser.parse_args()
    return args


def evaluation(corrects, outputs, labels):
    with torch.no_grad():
        leaf_labels = torch.nonzero(labels > 2)  # CUB2011 class: 122-321; HRSC2016/FGSC class: 3-23
        if leaf_labels.shape[0] > 0:
            select_leaf_labels = torch.index_select(labels, 0, leaf_labels.squeeze()) - 3
            select_pr = torch.index_select(outputs, 0, leaf_labels.squeeze())
            batch_pr = select_pr.data
            pr = batch_pr[:, 3:]
            _, predictions = torch.max(pr.data, 1)
            corrects += torch.sum(predictions == select_leaf_labels.data)
        else:
            corrects += 0
    return corrects


if __name__ == '__main__':
    args = arg_parse()
    # Running device
    device = torch.device("cuda:" + args.device)

    ## CUB2011
    # data_dir = '/home/datasets/HI_Datasets/Caltech-UCSD Birds-200-2011/CUB_200_2011/images'
    # train_list = '/home/datasets/HI_Datasets/Caltech-UCSD Birds-200-2011/CUB_200_2011/hierarchy/train_images_4_level_V1.txt'
    # test_list = '/home/datasets/HI_Datasets/Caltech-UCSD Birds-200-2011/CUB_200_2011/hierarchy/test_images_4_level_V1.txt'
    # data_transforms = makeDefaultTransforms()
    # CUB: All 4 Hierarchy
    # total_nodes = 372
    # CUB: order & class 2 Hierarchy
    # total_nodes = 213
    # CUB: genus & class 2 Hierarchy
    # total_nodes = 322

    ## HRSC2016
    train_dir = '/home/datasets/HI_Datasets/HRSC2016_cropped/Train'
    test_dir = '/home/datasets/HI_Datasets/HRSC2016_cropped/Test'

    ## FGSC
    # train_dir = '/home/datasets/HI_Datasets/FGSC-23/train'
    # test_dir = '/home/datasets/HI_Datasets/FGSC-23/test'

    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    total_nodes = 24

    batch_size = args.batch
    epochs = args.epoch
    num_workers = args.worker

    # CUB2011
    # train_set = CubDataset(data_dir, train_list, data_transforms['train'], 'genus', args.proportion)
    # test_set = CubDataset(data_dir, test_list, data_transforms['test'], 'class', args.proportion)

    # HRSC2016 & FGSC
    train_set = HRSCDataset(train_dir, image_transforms['train'], args.proportion)
    test_set = HRSCDataset(test_dir, image_transforms['test'], 1.0)

    train_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_data_size = len(train_set)
    test_data_size = len(test_set)

    ## Define Network
    # Start from scratch
    resnet50 = models.resnet50(pretrained=False)
    resnet50.load_state_dict(torch.load('./pre-trained/resnet50-19c8e357.pth'))
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(256, total_nodes),
                                nn.LogSoftmax(dim=1))
    # Start from previous epoch
    # resnet50 = torch.load(args.model)

    resnet50 = resnet50.to(device)

    cross_entropy = nn.NLLLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=args.lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    writer = SummaryWriter('runs_softmax_all_' + str(args.proportion))

    # Training loop
    best_acc = 0.0  # best accuracy on the test set
    best_epoch = 0  # corresponding epoch of the best accuracy
    best_model_wts = copy.deepcopy(resnet50.state_dict())

    for epoch in range(epochs):
        epoch_start = time.time()
        print('Epoch: {}/{}'.format(epoch, epochs) + ' Learning rate: {}'.format(scheduler.get_lr()[0]))

        resnet50.train()

        train_loss = 0.0
        train_acc = 0.0
        train_corrects = 0
        test_loss = 0.0
        test_acc = 0.0
        test_corrects = 0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = resnet50(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects = evaluation(train_corrects, outputs, labels)

        scheduler.step()

        # Evaluating on the test set in this epoch
        with torch.no_grad():
            resnet50.eval()
            for j, (inputs, labels) in enumerate(test_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = resnet50(inputs)
                loss = cross_entropy(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_corrects = evaluation(test_corrects, outputs, labels)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_corrects.double() / (train_data_size * args.proportion)
        # avg_train_acc = 0.0
        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_corrects.double() / test_data_size

        writer.add_scalars('loss', {
            'training loss': avg_train_loss,
            'testing loss': avg_test_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'Training Acc': avg_train_acc,
            'Testing Acc': avg_test_acc
        }, epoch)

        if best_acc < avg_test_acc:
            best_acc = avg_test_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(resnet50.state_dict())

        epoch_end = time.time()
        print('Epoch: {:03d}, Training Loss: {:.4f}, Training Accuracy: {:.4f}%, Time: {:.4f}s'.format(epoch,
                                                                                                       avg_train_loss,
                                                                                                       avg_train_acc * 100,
                                                                                                       epoch_end - epoch_start))
        print('Epoch: {:03d}, Test Loss: {:.4f}, Test Accuracy: {:.4f}%'.format(epoch,
                                                                                avg_test_loss,
                                                                                avg_test_acc * 100))
    writer.close()
    print('Best Accuracy on the test set: {:.4f}% at epoch: {:03d}'.format(best_acc * 100,
                                                                           best_epoch))
    resnet50.load_state_dict(best_model_wts)
    torch.save(resnet50, './softmax_all/HRSC_model_softmax_all_30-40_' + str(args.lr) + '_' + str(batch_size) + '_' + str(args.proportion) + '.pt')

    # num_classes = 21
    #
    # ## HRSC2016
    # # class_names = [100000005, 100000006, 100000007, 100000008, 100000009, 100000010, 100000011, 100000013, 100000015,
    # #                100000016, 100000018,
    # #                100000019, 100000020, 100000022, 100000024, 100000025, 100000026, 100000028, 100000029,
    # #                100000030, 100000032]
    # ## FGSC
    # # class_names = [1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 3, 4, 5, 6, 7, 8, 9]
    #
    # correct = 0
    # total = 0
    # # Accuracy of each class
    # class_correct = list(0. for i in range(num_classes))
    # class_total = list(0. for i in range(num_classes))
    # # Confusion matrix
    # y_true = []
    # y_pred = []
    # with torch.no_grad():
    #     resnet50.eval()
    #     for j, (images, labels) in enumerate(test_data_loader):
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         leaf_labels = labels - 3  # CUB2011 class: 122-321; HRSC2016 class: 3-23
    #         fc_total = resnet50(images)
    #         batch_pMargin = fc_total.data
    #         pMargin = batch_pMargin[:, 3:]  # CUB2011 class: 122-321; HRSC2016 class: 3-23
    #         _, predicted = torch.max(pMargin.data, 1)
    #         total += leaf_labels.size(0)
    #         correct += (predicted == leaf_labels).sum().item()
    #         c = (predicted == leaf_labels).squeeze()
    #         for i in range(leaf_labels.size(0)):
    #             label = leaf_labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    #             y_true.append(class_names[leaf_labels[i]])
    #             y_pred.append(class_names[predicted[i]])
    # print('Overall Accuracy of the trained network on the test set: {:.2f}%'.format(100 * correct / total))
    # for i in range(num_classes):
    #     print('Accuracy of {} : {:.2f}%'.format(class_names[i], 100 * class_correct[i] / class_total[i]))
