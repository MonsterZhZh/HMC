import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import scipy.io
import os
import random
import numpy as np
import argparse

from dataset import CubDataset, HRSCDataset, HRSCDataset2, DataHieAug
from transforms import makeDefaultTransforms
from hex_loss_batch import convert_to_torch
from hex_loss_batch import HEXLoss
from train_test import net_train, net_test, test_AP
from ResNetEmbed import ResNetEmbed

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--balance', default=1.0, type=float,
                        help='balance between softmax and hex (default: 1.0)')
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
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--mode', type=int, default=0,
                        help='HMC Test Modes (default: 0)')
    parser.add_argument('--dataset', type=str, default='HRSC', required=True,
                        help='dataset name')
    args = parser.parse_args()
    return args

def deploy(hexG, Eh, data_dir, train_list, test_list, data_transforms, ResNetEmbed,
           leaf_nodes, total_nodes, batch_size, num_workers, epochs, balance, device):
    # NxM Cell in Mat = NxM np.array(dtype=object)
    cliques = hexG['cliques'][0, 0]
    convert_to_torch(cliques, device)
    stateSpace = hexG['stateSpace'][0, 0]
    convert_to_torch(stateSpace, device)
    variables = hexG['variables'][0, 0]
    convert_to_torch(variables, device)
    childVariables = hexG['childVariables'][0, 0]
    convert_to_torch(childVariables, device)
    sumProduct = hexG['sumProduct'][0, 0]
    for i in range(np.size(sumProduct, 0)):
        convert_to_torch(sumProduct[i, 0], device)
    varTable = hexG['varTable'][0, 0]
    for i in range(np.size(varTable, 0)):
        convert_to_torch(varTable[i, 0], device)
    upMsgTable = hexG['upMsgTable'][0, 0]
    convert_to_torch(upMsgTable, device)
    downMsgTable = hexG['downMsgTable'][0, 0]
    convert_to_torch(downMsgTable, device)
    numVar = hexG['numVar'][0, 0]
    numVar = torch.from_numpy(numVar.astype(np.int64)).to(torch.long).to(device)
    cliqParents = hexG['cliqParents'][0, 0]
    cliqParents = torch.from_numpy(cliqParents.astype(np.int64)).to(torch.long).to(device)
    upPass = hexG['upPass'][0, 0]
    upPass = torch.from_numpy(upPass.astype(np.int64)).to(torch.long).to(device)
    Eh = torch.from_numpy(Eh).to(torch.long).to(device)

    ## CUB2011
    # train_set = CubDataset(data_dir, train_list, data_transforms['train'], 'genus', args.proportion)
    # test_set = CubDataset(data_dir, test_list, data_transforms['test'], 'class', 1.0)

    ## HRSC2016 & FGSC
    train_set = HRSCDataset(train_list, args.dataset, data_transforms['train'], args.proportion)
    test_set = HRSCDataset(test_list, args.dataset, data_transforms['test'], 1.0)

    # train_set = DataHieAug(train_list, args.dataset, data_transforms['train'], training=True)
    # test_set = DataHieAug(test_list, args.dataset, data_transforms['test'], training=False)

    # test_set = HRSCDataset2(test_list, args.dataset, data_transforms['test'], 1.0)

    train_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_data_size = len(train_set)
    test_data_size = len(test_set)

    ## Define Network
    # Start from scratch
    # ResNetEmbed = ResNetEmbed(leaf_nodes, total_nodes)
    # Start from previous epoch
    # ResNetEmbed = torch.load(args.model)
    ResNetEmbed = torch.load(args.model, map_location='cpu')

    ResNetEmbed = ResNetEmbed.to(device)

    # Customized HEX Loss
    hex = HEXLoss(cliques, stateSpace, numVar, cliqParents, childVariables, upPass,
                  sumProduct, upMsgTable, downMsgTable, variables, varTable, Eh)
    # Cross-entropy loss
    cross_entropy = nn.NLLLoss()

    optimizer = optim.SGD(ResNetEmbed.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Define network and train it
    # model = net_train(ResNetEmbed, hex, cross_entropy, optimizer, scheduler, balance, epochs,
    #                   device, train_data_loader, test_data_loader, train_data_size, test_data_size, args.proportion)
    # torch.save(model, './models/' + str(args.dataset) + '_model_HEX_20-40_0.001_' + str(balance) + '_' + str(batch_size) + '_' + str(args.proportion) + '.pt')

    # Test network on the test set
    # CUB2011
    # class_names = []
    # with open('/home/datasets/HI_Datasets/Caltech-UCSD Birds-200-2011/CUB_200_2011/classes.txt', 'r') as f:
    #     for l in f.readlines():
    #         number, class_name = l.strip().split(' ')
    #         class_names.append(class_name)
    # net_test(ResNetEmbed, device, test_data_loader, 200, class_names, hex, balance)

    if args.dataset == 'HRSC':
        class_names = [100000005, 100000006, 100000007, 100000008, 100000009, 100000010, 100000011, 100000013, 100000015,
                       100000016, 100000018,
                       100000019, 100000020, 100000022, 100000024, 100000025, 100000026, 100000028, 100000029,
                       100000030, 100000032]
    elif args.dataset == 'FGSC':
        class_names = [1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 3, 4, 5, 6, 7, 8, 9]

    net_test(ResNetEmbed, device, test_data_loader, 21, class_names, hex, balance)
    # test_AP(ResNetEmbed, device, test_set, test_data_loader, balance, hex, args.mode)



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
    # leaf_nodes = 200

    if args.dataset == 'HRSC':
        data_dir = ''
        train_dir = '/home/datasets/HI_Datasets/HRSC2016_cropped/Train'
        test_dir = '/home/datasets/HI_Datasets/HRSC2016_cropped/Test'
        hexG = scipy.io.loadmat("./HEX/hexG_HRSC.mat")['hexG']
        Eh = scipy.io.loadmat("./HEX/Subsumption_HRSC.mat")['Eh']
    elif args.dataset == 'FGSC':
        data_dir = ''
        train_dir = '/home/datasets/HI_Datasets/FGSC-23/train'
        test_dir = '/home/datasets/HI_Datasets/FGSC-23/test'
        hexG = scipy.io.loadmat("./HEX/hexG_FGSC.mat")['hexG']
        Eh = scipy.io.loadmat("./HEX/Subsumption_FGSC.mat")['Eh']

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
    leaf_nodes = 21

    batch_size = args.batch
    epochs = args.epoch
    balance = args.balance
    num_workers = args.worker
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## CUB2011
    # deploy(hexG, Eh, data_dir, train_list, test_list, data_transforms, ResNetEmbed,
    #        leaf_nodes, total_nodes, batch_size, num_workers, epochs, balance, device)

    ## HRSC2016
    deploy(hexG, Eh, data_dir, train_dir, test_dir, image_transforms, ResNetEmbed,
           leaf_nodes, total_nodes, batch_size, num_workers, epochs, balance, device)
