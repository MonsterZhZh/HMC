import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pickle
import cv2
from os.path import join


def display_HMC_results(results_txt, save_path):
    number = 0
    HRSC_hierarchy = {'0': ['0', '1', '7', '9', '20'],
                      '1': ['2', '3', '4', '5', '6', '8', '11', '17', '18'],
                      '2': ['10', '12', '13', '14', '15', '16', '19']}
    name_list = ('AC', 'WC', 'MS',
                 'NT', 'ET', 'AB', 'WI', 'OHP', 'SA', 'TD', 'KT', 'AT', 'TA', 'CTS', 'CS', 'CC-A', 'HC', 'YT', 'CGS', 'CUS', 'WCA', 'MDS', 'CC-B', 'MD')
    num_list_sof = []
    num_list_sig = []
    with open(results_txt, "r") as f:
        for line in f.readlines():
            line = line.strip('\n').strip('(').strip(')').strip('[').strip(']').strip('[').strip(']').strip(',')

            if not line:
                number += 1
            else:
                if (number % 4) == 0:
                    lines = line.split('/')
                    img_name = lines[-1].strip('\'')
                elif (number % 4) == 1:
                    num_list_GT = [0 for i in range(24)]
                    if line in HRSC_hierarchy['0']:
                        father_label = 0
                    elif line in HRSC_hierarchy['1']:
                        father_label = 1
                    elif line in HRSC_hierarchy['2']:
                        father_label = 2
                    leaf_label = int(line) + 3
                    num_list_GT[father_label] = 1
                    num_list_GT[leaf_label] = 1
                elif (number % 4) == 2:
                    lines = line.strip().split()
                    for i in lines:
                        num_list_sof.append(float(i))
                elif (number % 4) == 3:
                    lines = line.strip().split()
                    for i in lines:
                        num_list_sig.append(float(i))

            if len(num_list_sof) == 21 and len(num_list_sig) == 24:
                n_groups = 24 * 3
                fig, ax = plt.subplots()
                index = np.arange(0, n_groups, 3)
                bar_width = 0.8
                tick_width = 0.8
                opacity = 1.0
                error_config = {'ecolor': '0.3'}

                rects1 = ax.barh(index, num_list_sig, bar_width, alpha=opacity, color='b', error_kw=error_config,
                                 label='Softmax Outputs')
                rects2 = ax.barh(index + bar_width * 1, num_list_sig, bar_width, alpha=opacity, color='c',
                                 error_kw=error_config,
                                 label='Sigmoid Outputs')
                rects3 = ax.barh(index + bar_width * 2, num_list_GT, bar_width, alpha=opacity, color='r', error_kw=error_config,
                                 label='GT')

                ax.set_xlabel('Prediction Scores')
                ax.set_ylabel('All Labels in the Hierarchy')
                ax.set_title('Comparison of Coherent Predictions')
                # ax.set_xticks(index + bar_width / 2)
                # ax.set_xticklabels(name_list_HRSC)
                ax.set_yticks(index + tick_width)
                ax.set_yticklabels(name_list)
                ax.legend()

                fig.tight_layout()
                plt.savefig(save_path + img_name + '.jpg', dpi=500)
                # plt.show()
                plt.close()

                num_list_sof = []
                num_list_sig = []


def filterImgs(source_path, dest_path):
    '''
    remove images from dest_path according to source_path
    '''
    source_lists = os.listdir(source_path)
    dest_lists = os.listdir(dest_path)
    source_img_names = []
    dest_img_names = []
    not_include = []
    for source in source_lists:
        img_name = source.split('.')[0]
        source_img_names.append(img_name)
    for dest in dest_lists:
        img_name = dest.split('.')[0]
        dest_img_names.append(img_name)
    for img_name in dest_img_names:
        if img_name not in source_img_names:
            not_include.append(img_name)
            file = img_name + '.jpg'
            if os.path.isfile(os.path.join(dest_path, file)):
                os.remove(os.path.join(dest_path, file))
    return not_include


def DownSamplingImgs(source_path, dest_path):
    source_lists = os.listdir(source_path)
    for source in source_lists:
        img_path = join(source_path, source)
        img = cv2.imread(img_path)
        # w = img.shape[1]
        # h = img.shape[0]
        new_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(join(dest_path, source), new_img)


def map_OD_HMC(OD_path, HMC_path):
    OD_lists = os.listdir(OD_path)
    HMC_lists = os.listdir(HMC_path)
    OD_img_names = []
    HMC_img_names = []
    map = {}
    for OD in OD_lists:
        img_name = OD.split('.')[0]
        OD_img_names.append(img_name)
        map[img_name] = []
    for HMC in HMC_lists:
        img_name = HMC.split('.')[0]
        HMC_img_names.append(img_name)
    for HMC_img in HMC_img_names:
        for OD_img in OD_img_names:
            if HMC_img.startswith(OD_img):
                map[OD_img].append(HMC_img)
                break
    return map

if __name__ == '__main__':
    source_path = 'F:\\HI_Datasets\\FGSC-23\\train\\8'
    dest_path = 'C:\\Users\\chenj\\Desktop\\2\\8'
    DownSamplingImgs(source_path, dest_path)
    # not_include = filterImgs(source_path, dest_path)
    # print(not_include)

    # OD_path = 'I:\\HRSC\\OD\\'
    # HMC_path = 'I:\\HRSC\\HMC\\'
    # map = map_OD_HMC(OD_path, HMC_path)
    # OD_HMC_Database = {}
    # with open('I:\\HRSC\\instances.pkl', 'rb') as f:
    #     instances = pickle.load(f)
    # for key in instances.keys():
    #     key2 = key.split('.')[0]
    #     if key2 in map.keys() and instances[key] == len(map[key2]):
    #         OD_HMC_Database[key2] = map[key2]
    # with open('I:\\HRSC\\OD_HMC_Database.pkl', 'wb') as f:
    #     pickle.dump(OD_HMC_Database, f, pickle.HIGHEST_PROTOCOL)
    # print('OK')