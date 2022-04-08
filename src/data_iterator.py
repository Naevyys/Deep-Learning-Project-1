import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomAffine
from torchvision.transforms.functional import affine


class RandomAffine2Img(RandomAffine):
    def __call__(self, img1, img2):
        """
        Applies the same random affine transformations to two images
        :param img1: First image
        :param img2: Second image
        :return: Tuple of transformed images
        """

        # Get random parameters
        angle, translations, scale, shear = self.get_params(self.degrees, self.translate, self.scale, None, img1.size())
        translations, shear = list(translations), list(shear)  # We need lists instead of tuples for the next step

        # Transform images
        transformed_img1 = affine(img1, angle, translations, scale, shear, fill=self.fill)
        transformed_img2 = affine(img2, angle, translations, scale, shear, fill=self.fill)
        return transformed_img1, transformed_img2


class DataIterator(Dataset):
    def __init__(self, imgs1, imgs2, degrees=0, translate=None, scale=None, fill=0):
        """
        Initializes a data iterator for our data.
        :param imgs1: Tensor of first images
        :param imgs2: Tensor of second images
        :param degrees: Int or tuple of values for degrees of rotation
        :param translate: Tuple of values for maximal translations
        :param scale: Tuple giving scaling factor interval
        :param fill: Pixel value for areas outside of transformed images
        :return: None
        """

        # Notes:
        # - No need to clean our data, since we have no useless or collinear features
        # - No need to balance our data, since we do not classify anything
        # - This iterator does not avoid loading everything into memory, since we first load the entire data, then
        #   pass it to the interator
        # - Do we want to add a color altering transformation to make our models more robust to pixel colors?

        assert imgs1.size() == imgs2.size(), "Tensors of first and second images must have the same size!"
        assert isinstance(imgs1, torch.Tensor) and isinstance(imgs2, torch.Tensor), "Images must be tensors!"
        assert isinstance(degrees, int) or isinstance(degrees, tuple) and len(degrees) == 2, \
            "Degrees must be an integer or a list of integers"
        assert isinstance(translate, tuple) and len(translate) == 2 or translate is None, \
            "Translate must be a tuple of size 2 or None"
        assert isinstance(scale, tuple) and len(scale) == 2 or scale is None, \
            "Scale must be a tuple of size 2 or None"
        assert isinstance(fill, int), "Fill value must be an integer"

        self.imgs1 = imgs1
        self.imgs2 = imgs2

        self.transformer = RandomAffine2Img(degrees, translate=translate, scale=scale, fill=fill)

    def __transform_images(self, index):
        """
        Applies required transformations to images at index passed. The same transformation is applied to both images.
        :param index: Index of requested images
        :return: Pair of augmented images
        """

        img1 = self.imgs1[index]
        img2 = self.imgs2[index]

        img1, img2 = self.transformer(img1, img2)  # Apply same transformation on both images

        return img1, img2

    def __getitem__(self, index):
        img1, img2 = self.__transform_images(index)
        return img1, img2

    def __len__(self):
        return len(self.imgs1)
