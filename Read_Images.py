import os
import numpy as np
import cv2
from Transforms_List import Seg_Transforms


class Read_Images():
    """
    Read the images used to train and validate and return them to calling function

    Args:
        img_height: height of image desired
        img_width: width of image desired
        pat_list: list of patients for the training and validation of algorithm
        train_dir: directory where patient training images can be found
        valid_dir: directory where patient validation images can be found

    """

    def __init__(self, img_height=500, img_width=500, sample_list=dict, train_dir="", valid_dir=""):
        self.img_height = img_height
        self.img_width = img_width
        self.sample_list = sample_list
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.transformImg, self.transformAnn = Seg_Transforms(self.img_height, self.img_width).standardTransforms()

    def Train_Reader(self):
        """
        Reads an image along with its dense and breast masks and creates an annotation map. These two items are then
        passed to a transform function before being returned to the calling function. This function is used for training
        batches.

        :return: transformed image and transformed annotation map
        """

        sample_idx = np.random.randint(0, len(self.sample_list["Train"]))
        sample_dir = self.train_dir + "/" + self.sample_list["Train"][sample_idx]
        views = os.listdir(sample_dir)
        view_idx = np.random.randint(0, len(views))
        view_dir = sample_dir + "/" + views[view_idx]
        for item in os.listdir(view_dir):
            if "breast" in item:
                class1 = cv2.imread(sample_dir + "/" + views[view_idx] + "/" + item)
                class1 = cv2.cvtColor(class1, cv2.COLOR_BGR2GRAY)
            if "dense" in item:
                class2 = cv2.imread(sample_dir + "/" + views[view_idx] + "/" + item)
                class2 = cv2.cvtColor(class2, cv2.COLOR_BGR2GRAY)
            if "breast" not in item and "dense" not in item:
                Img = cv2.imread(sample_dir + "/" + views[view_idx] + "/" + item)
        AnnMap = np.zeros(Img.shape[0:2], np.float32)
        if class1 is not None:  AnnMap[class1 == 255] = 1
        if class2 is not None: AnnMap[class2 == 255] = 2
        Img = self.transformImg(Img)
        AnnMap = self.transformAnn(AnnMap)
        return Img, AnnMap

    def Val_Reader(self):
        """
        Reads an image along with its dense and breast masks and creates an annotation map. These two items are then
        passed to a transform function before being returned to the calling function. This function is used for
        validation batchs.

        :return: transformed image and transformed annotation map
        """

        val_sample_idx = np.random.randint(0, len(self.sample_list["Validation"]))
        val_sample_dir = self.valid_dir + "/" + self.sample_list["Validation"][val_sample_idx]
        val_views = os.listdir(val_sample_dir)
        val_view_idx = np.random.randint(0, len(val_views))
        val_view_dir = val_sample_dir + "/" + val_views[val_view_idx]
        for item in os.listdir(val_view_dir):
            if "breast" in item:
                class1 = cv2.imread(val_sample_dir + "/" + val_views[val_view_idx] + "/" + item)
                class1 = cv2.cvtColor(class1, cv2.COLOR_BGR2GRAY)
            if "dense" in item:
                class2 = cv2.imread(val_sample_dir + "/" + val_views[val_view_idx] + "/" + item)
                class2 = cv2.cvtColor(class2, cv2.COLOR_BGR2GRAY)
            if "breast" not in item and "dense" not in item:
                Img = cv2.imread(val_sample_dir + "/" + val_views[val_view_idx] + "/" + item)
        AnnMap = np.zeros(Img.shape[0:2], np.float32)
        if class1 is not None:  AnnMap[class1 == 255] = 1
        if class2 is not None: AnnMap[class2 == 255] = 2
        Img = self.transformImg(Img)
        AnnMap = self.transformAnn(AnnMap)
        return Img, AnnMap
