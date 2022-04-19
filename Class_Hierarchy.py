from os.path import join
from scipy.sparse import lil_matrix
import scipy.io as io

if __name__ == "__main__":
    hierarchy_list = "/home/datasets/HI_Datasets/Caltech-UCSD Birds-200-2011/CUB_200_2011/hierarchy/train_images_4_level_V1.txt"
    image_dir = "/home/datasets/HI_Datasets/Caltech-UCSD Birds-200-2011/CUB_200_2011/hierarchy/images"
    save_path = "./HEX/Subsumption_CUB_genus_class"

    # Subsumption Matrix ordered by order (1-13), family (1-37) + 13, genus (1-122) + 50, class (1-200) + 172
    # CUB: All 4 Hierarchy
    # Eh = lil_matrix((372, 372), dtype=int)
    # CUB: order & class 2 Hierarchy
    # Eh = lil_matrix((213, 213), dtype=int)
    # CUB: genus & class 2 Hierarchy
    Eh = lil_matrix((322, 322), dtype=int)

    with open(hierarchy_list, 'r') as f:
        for line in f.readlines():
            image_name, class_label, genus_label, family_label, order_label = line.strip().split(' ')

            # CUB: All 4 Hierarchy
            # if Eh[int(order_label)-1, int(family_label)+12] == 0:
            #     Eh[int(order_label)-1, int(family_label)+12] = 1
            # if Eh[int(family_label)+12, int(genus_label)+49] == 0:
            #     Eh[int(family_label)+12, int(genus_label)+49] = 1
            # if Eh[int(genus_label)+49, int(class_label)+171] == 0:
            #     Eh[int(genus_label)+49, int(class_label)+171] = 1

            # CUB: order & class 2 Hierarchy
            # if Eh[int(order_label) - 1, int(class_label) + 12] == 0:
            #     Eh[int(order_label) - 1, int(class_label) + 12] = 1

            # CUB: genus & class 2 Hierarchy
            if Eh[int(genus_label) - 1, int(class_label) + 121] == 0:
                Eh[int(genus_label) - 1, int(class_label) + 121] = 1

    io.savemat(save_path, {'Eh': Eh.toarray()})