import json
import random
import os

dir = "./jpeg_segmentations"

patients = os.listdir(dir)

random.Random(42).shuffle(patients)

block_size = int(len(patients)/5)

data_dict = {"CV1":{"Train":[], "Validation":[], "Test":[]},
             "CV2":{"Train":[], "Validation":[], "Test":[]},
             "CV3":{"Train":[], "Validation":[], "Test":[]},
             "CV4":{"Train":[], "Validation":[], "Test":[]},
             "CV5":{"Train":[], "Validation":[], "Test":[]}}

block_1 = patients[:block_size]
block_2 = patients[block_size:block_size*2]
block_3 = patients[block_size*2:block_size*3]
block_4 = patients[block_size*3:block_size*4]
block_5 = patients[block_size*4:]

blocks = [block_1, block_2, block_3, block_4, block_5]

test = 4
validation = 3

for fold in data_dict:
    for index, i in enumerate(blocks):
        if index == validation:
            data_dict[fold]["Validation"].extend(blocks[int(index)])
        if index == test:
            data_dict[fold]["Test"].extend(blocks[int(index)])
        elif index != validation and index != test:
            data_dict[fold]["Train"].extend(blocks[int(index)])
    test-=1
    validation-=1
    if test < 0:
        test = 4
    if validation < 0:
        validation = 4

# for fold in data_dict:
#     print(len(data_dict[fold]["Train"]), len(data_dict[fold]["Validation"]), len(data_dict[fold]["Test"]))


with open("Segmentation_Folds.json", "w") as g:
    json.dump(data_dict, g, indent = 3)


