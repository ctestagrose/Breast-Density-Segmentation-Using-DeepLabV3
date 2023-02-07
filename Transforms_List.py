import torchvision.transforms as tf


class Seg_Transforms():
    """
    Transform list for use with the segmentation algorithm.
    Evaluation will use the transformImgOnly method while the training and validation procedure will use standardTransforms
    Additional functionality can be added if more transforms are desired.

    Args:
        img_height: height of the image
        img_width: width of the image

    """

    def __init__(self, img_height=int, img_width=int):
        self.img_height = img_height
        self.img_width = img_width

    def standardTransforms(self):
        """
        Applies a standard list of transforms to both the images and annotation maps

        :return: returns a transformed image and its corresponding transformed annotationmap
        """

        transformImg = tf.Compose(
            [tf.ToPILImage(), tf.Resize((self.img_height, self.img_width), interpolation=tf.InterpolationMode.BILINEAR),
             tf.ToTensor(),
             tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transformAnn = tf.Compose(
            [tf.ToPILImage(), tf.Resize((self.img_height, self.img_width), interpolation=tf.InterpolationMode.NEAREST),
             tf.ToTensor()])

        return transformImg, transformAnn

    def transformImgOnly(self):
        """
        Applies a list of transforms to an image only this is for evaluation and future deployment

        :return: a tranformed image ready for input in the model
        """

        transformImg = tf.Compose(
            [tf.ToPILImage(), tf.Resize((self.img_height, self.img_width), interpolation=tf.InterpolationMode.BILINEAR),
             tf.ToTensor(),
             tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        return transformImg
