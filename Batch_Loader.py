import torch
from Read_Images import Read_Images


class Loader():
    """
    Load batch for training and validation

    Args:
        img_height: height of image desired
        img_width: width of image desired
        batch_size: the batch size used for training
        pat_list: list of patients for the training and validation of algorithm
        train_dir: directory where patient training images can be found
        valid_dir: directory where patient validation images can be found
    """

    def __init__(self, img_height=500, img_width=500, batch_size=2, pat_list=dict, train_dir="",
                 valid_dir=""):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.pat_list = pat_list
        self.train_dir = train_dir
        self.valid_dir = valid_dir

    def Train_loader(self):
        """
        Loads training batch for both images and annotation maps

        :return:image and annotation batches
        """

        images = torch.zeros([self.batch_size, 3, self.img_height, self.img_width])
        ann = torch.zeros([self.batch_size, self.img_height, self.img_width])

        for i in range(self.batch_size):
            images[i], ann[i] = Read_Images(self.img_height,
                                            self.img_width,
                                            self.pat_list,
                                            self.train_dir,
                                            self.valid_dir).Train_Reader()

        return images, ann

    def Validation_loader(self):
        """
        Loads validation batch for both images and annotation maps

        :return:image and annotation batches
        """
        images = torch.zeros([self.batch_size, 3, self.img_height, self.img_width])
        ann = torch.zeros([self.batch_size, self.img_height, self.img_width])

        for i in range(self.batch_size):
            images[i], ann[i] = Read_Images(self.img_height,
                                            self.img_width,
                                            self.pat_list,
                                            self.train_dir,
                                            self.valid_dir).Val_Reader()

        return images, ann
