import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

from os.path import join
from PIL import Image
import random
import math
import os
import networkx as nx
import numpy as np


class CubDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(CubDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion

        name_list = []
        order_label_list = []
        family_label_list = []
        genus_label_list = []
        class_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, class_label, genus_label, family_label, order_label = l.strip().split(' ')
                name_list.append(imagename)
                order_label_list.append(int(order_label))
                family_label_list.append(int(family_label) + 13)
                # genus_label_list.append(int(genus_label) + 50)
                # class_label_list.append(int(class_label) + 172)
                genus_label_list.append(int(genus_label))
                class_label_list.append(int(class_label) + 122)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, order_label_list, family_label_list, genus_label_list, class_label_list)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index] - 1

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def relabel(self, image_filenames, order_label_list, family_label_list, genus_label_list, class_label_list):
        class_imgs = {}
        for i in range(len(image_filenames)):
            if str(class_label_list[i]) not in class_imgs.keys():
                class_imgs[str(class_label_list[i])] = {'images': [], 'order': [], 'family': [], 'genus': []}
                class_imgs[str(class_label_list[i])]['images'].append(image_filenames[i])
                class_imgs[str(class_label_list[i])]['order'].append(order_label_list[i])
                class_imgs[str(class_label_list[i])]['family'].append(family_label_list[i])
                class_imgs[str(class_label_list[i])]['genus'].append(genus_label_list[i])
            else:
                class_imgs[str(class_label_list[i])]['images'].append(image_filenames[i])
        labels = []
        images = []
        for key in class_imgs.keys():
            # random.shuffle(class_imgs[key]['images'])
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            print(key + ' has the rest: ' + str(rest))
            if self.re_level == 'order':
                labels += class_imgs[key]['order'] * rest
            elif self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'genus':
                labels += class_imgs[key]['genus'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')

        return images, labels


class HRSCDataset(data.Dataset):
    def __init__(self, image_dir, dataset, input_transform=None, proportion=1.0):
        super(HRSCDataset, self).__init__()
        self.input_transform = input_transform
        self.proportion = proportion
        if dataset == 'HRSC':
            self.hierarchy = {'0': ['100000005', '100000006', '100000013', '100000016', '100000032'],
                              '1': ['100000007', '100000008', '100000009', '100000010', '100000011',
                                    '100000015', '100000019', '100000028', '100000029'],
                              '2': ['100000018', '100000020', '100000022', '100000024', '100000025',
                                    '100000026', '100000030']}
        elif dataset == 'FGSC':
            self.hierarchy = {'0': ['1', '8'],
                              '1': ['2', '3', '4', '5', '6', '7', '9', '11', '12', '13'],
                              '2': ['14', '15', '16', '17', '18', '19', '20', '21', '22']}

        name_list = []
        label_list = []
        classes = os.listdir(image_dir)
        classes = sorted(classes)
        i = 3
        for cls in classes:
            tmp_name_list = []
            tmp_class_label_list = []
            if dataset == 'HRSC' and cls == '100000027':
                continue
            if dataset == 'FGSC' and cls in ['0', '10']:
                continue
            cls_imgs = join(image_dir, cls)
            imgs = os.listdir(cls_imgs)
            for img in imgs:
                tmp_name_list.append(join(image_dir, cls, img))
                tmp_class_label_list.append(i)
            i += 1
            name_list += tmp_name_list
            label_list += tmp_class_label_list[:math.ceil(len(tmp_class_label_list) * self.proportion)]
            rest = len(tmp_class_label_list) - math.ceil(len(tmp_class_label_list) * self.proportion)
            if cls in self.hierarchy['0']:
                label_list += [0] * rest
            elif cls in self.hierarchy['1']:
                label_list += [1] * rest
            elif cls in self.hierarchy['2']:
                label_list += [2] * rest
        self.image_filenames = name_list
        self.label_list = label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        label = self.label_list[index]
        if label < 3:
            # DownSampling for samples labeled as coarse-grained
            loader = transforms.Compose([transforms.ToTensor()])
            unloader = transforms.ToPILImage()
            input = loader(input).unsqueeze(0)
            input = F.interpolate(input, scale_factor=0.5)
            input = input.squeeze(0)
            input = unloader(input)
        if self.input_transform:
            input = self.input_transform(input)
        return input, label

    def __len__(self):
        return len(self.image_filenames)


to_skip = ['root']


class HRSCDataset2(data.Dataset):
    def __init__(self, image_dir, dataset, input_transform=None, proportion=1.0):
        super(HRSCDataset2, self).__init__()
        self.input_transform = input_transform
        self.proportion = proportion

        if dataset == 'HRSC':
            self.hierarchy = {'100000002': ['100000005', '100000006', '100000013', '100000016', '100000032'],
                              '100000003': ['100000007', '100000008', '100000009', '100000010', '100000011',
                                            '100000015', '100000019', '100000028', '100000029'],
                              '100000004': ['100000018', '100000020', '100000022', '100000024', '100000025',
                                            '100000026', '100000030']}
        elif dataset == 'FGSC':
            self.hierarchy = {'00': ['1', '8'],
                              '01': ['2', '3', '4', '5', '6', '7', '9', '11', '12', '13'],
                              '02': ['14', '15', '16', '17', '18', '19', '20', '21', '22']}

        self.g, self.g_t, self.adj_matrix, self.to_eval, self.nodes_idx = self.compute_adj_matrix()

        name_list = []
        label_list = []
        classes = os.listdir(image_dir)
        classes = sorted(classes)

        for cls in classes:
            tmp_name_list = []
            tmp_class_label_list = []
            if dataset == 'HRSC' and cls == '100000027':
                continue
            if dataset == 'FGSC' and cls in ['0', '10']:
                continue
            cls_imgs = join(image_dir, cls)
            imgs = os.listdir(cls_imgs)
            y_ = np.zeros(25)
            y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, cls)]] = 1
            y_[self.nodes_idx[cls]] = 1
            for img in imgs:
                tmp_name_list.append(join(image_dir, cls, img))
                tmp_class_label_list.append(y_)

            name_list += tmp_name_list
            label_list += tmp_class_label_list[:int(math.ceil(len(tmp_class_label_list) * self.proportion))]
            rest = len(tmp_class_label_list) - math.ceil(len(tmp_class_label_list) * self.proportion)
            y_ = np.zeros(25)

            if dataset == 'HRSC':
                if cls in self.hierarchy['100000002']:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, '100000002')]] = 1
                    y_[self.nodes_idx['100000002']] = 1
                elif cls in self.hierarchy['100000003']:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, '100000003')]] = 1
                    y_[self.nodes_idx['100000003']] = 1
                elif cls in self.hierarchy['100000004']:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, '100000004')]] = 1
                    y_[self.nodes_idx['100000004']] = 1
            elif dataset == 'FGSC':
                if cls in self.hierarchy['00']:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, '00')]] = 1
                    y_[self.nodes_idx['00']] = 1
                elif cls in self.hierarchy['01']:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, '01')]] = 1
                    y_[self.nodes_idx['01']] = 1
                elif cls in self.hierarchy['02']:
                    y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, '02')]] = 1
                    y_[self.nodes_idx['02']] = 1

            label_list += [y_] * int(rest)
        self.image_filenames = name_list
        self.label_list = label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        label = self.label_list[index]
        return input, label

    def __len__(self):
        return len(self.image_filenames)

    def compute_adj_matrix(self):
        g = nx.DiGraph()
        for key in self.hierarchy.keys():
            g.add_edge(key, 'root')
            for item in self.hierarchy[key]:
                g.add_edge(item, key)
        nodes = sorted(g.nodes())
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        g_t = g.reverse()
        am = nx.to_numpy_matrix(g, nodelist=nodes, order=nodes)
        to_eval = [t not in to_skip for t in nodes]
        return g, g_t, np.array(am), to_eval, nodes_idx


class DataHieAug(data.Dataset):

    def __init__(self, image_dir, dataset, input_transform=None, training=True):
        super(DataHieAug, self).__init__()
        self.input_transform = input_transform
        if dataset == 'HRSC':
            self.hierarchy = {'0': ['100000005', '100000006', '100000013', '100000016', '100000032'],
                              '1': ['100000007', '100000008', '100000009', '100000010', '100000011',
                                    '100000015', '100000019', '100000028', '100000029'],
                              '2': ['100000018', '100000020', '100000022', '100000024', '100000025',
                                    '100000026', '100000030']}
        elif dataset == 'FGSC':
            self.hierarchy = {'0': ['1', '8'],
                              '1': ['2', '3', '4', '5', '6', '7', '9', '11', '12', '13'],
                              '2': ['14', '15', '16', '17', '18', '19', '20', '21', '22']}

        name_list = []
        label_list = []
        classes = os.listdir(image_dir)
        classes = sorted(classes)
        i = 3
        for cls in classes:
            tmp_name_list = []
            tmp_class_label_list = []
            if dataset == 'HRSC' and cls == '100000027':
                continue
            if dataset == 'FGSC' and cls in ['0', '10']:
                continue
            cls_imgs = join(image_dir, cls)
            imgs = os.listdir(cls_imgs)
            for img in imgs:
                tmp_name_list.append(join(image_dir, cls, img))
                tmp_class_label_list.append(i)
                if training:
                    tmp_name_list.append(join(image_dir, cls, img))
                    if cls in self.hierarchy['0']:
                        tmp_class_label_list.append(0)
                    elif cls in self.hierarchy['1']:
                        tmp_class_label_list.append(1)
                    elif cls in self.hierarchy['2']:
                        tmp_class_label_list.append(2)
            i += 1
            name_list += tmp_name_list
            label_list += tmp_class_label_list
        self.image_filenames = name_list
        self.label_list = label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        label = self.label_list[index]
        if label < 3:
            loader = transforms.Compose([transforms.ToTensor()])
            unloader = transforms.ToPILImage()
            input = loader(input).unsqueeze(0)
            input = F.interpolate(input, scale_factor=0.5)
            input = input.squeeze(0)
            input = unloader(input)
        if self.input_transform:
            input = self.input_transform(input)
        return input, label

    def __len__(self):
        return len(self.image_filenames)