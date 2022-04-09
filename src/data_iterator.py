import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomAffine, ColorJitter
from torchvision.transforms.functional import affine, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


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


class ColorJitter2Img(ColorJitter):
    def __call__(self, img1, img2):
        """
        Applies the same random color transformations to two images
        :param img1: First image
        :param img2: Second image
        :return: Tuple of transformed images
        """

        # Get random parameters
        order, brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        # Transform images
        transformed_img1 = img1
        transformed_img2 = img2
        for i in order:
            if i == 0 and brightness is not None:
                transformed_img1 = adjust_brightness(transformed_img1, brightness)
                transformed_img2 = adjust_brightness(transformed_img2, brightness)
            elif i == 1 and contrast is not None:
                transformed_img1 = adjust_contrast(transformed_img1, contrast)
                transformed_img2 = adjust_contrast(transformed_img2, contrast)
            elif i == 2 and saturation is not None:
                transformed_img1 = adjust_saturation(transformed_img1, saturation)
                transformed_img2 = adjust_saturation(transformed_img2, saturation)
            elif i == 3 and hue is not None:
                transformed_img1 = adjust_hue(transformed_img1, hue)
                transformed_img2 = adjust_hue(transformed_img2, hue)

        return transformed_img1, transformed_img2


class DataIterator(Dataset):
    def __init__(self, imgs1, imgs2, degrees=0, translate=None, scale=None, fill=0, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        """
        Initializes a data iterator for our data.
        :param imgs1: Tensor of first images.
        :param imgs2: Tensor of second images.
        :param degrees: Int or tuple of ints (range) for degrees of rotation.
        :param translate: Tuple of values (range) for maximal translations. Values between 0 and 1.
        :param scale: Tuple (range) giving scaling factor interval.
        :param fill: Pixel value for areas outside transformed images.
        :param brightness: Float or tuple of floats (range). Non-negative.
        :param contrast: Float or tuple of floats (range). Non-negative.
        :param saturation: Float or tuple of floats (range). Non-negative.
        :param hue: Float or tuple of floats (range). Float between 0 and 0.5 or tuple with values between -0.5 and 0.5.
        :return: None
        """

        # Notes:
        # - No need to clean our data, since we have no useless or collinear features, or at most maybe color channels
        #   -> TODO: investigate by training on black and white images as well
        # - No need to balance our data, since we do not classify anything
        # - This iterator does not avoid loading everything into memory, since we first load the entire data, then
        #   pass it to the interator
        # - We do not shear our images as they have a very low resolution

        assert isinstance(imgs1, torch.Tensor) and isinstance(imgs2, torch.Tensor), "Images must be tensors!"
        assert imgs1.size() == imgs2.size(), "Tensors of first and second images must have the same size!"
        assert isinstance(degrees, int) or isinstance(degrees, tuple) and len(degrees) == 2 \
            and isinstance(degrees[0], int) and isinstance(degrees[1], int), \
            "Degrees must be an integer or a list of integers"
        assert isinstance(translate, tuple) and len(translate) == 2 and 0 <= translate[0] <= 1 \
               and 0 <= translate[1] <= 1 or translate is None, "Translate must be a tuple of size 2 with values in " \
                                                                "[0, 1] or None"
        assert isinstance(scale, tuple) and len(scale) == 2 or scale is None, \
            "Scale must be a tuple of size 2 or None"
        assert isinstance(fill, int), "Fill value must be an integer"
        assert isinstance(brightness, tuple) and brightness[0] >= 0 and brightness[1] >= 0 \
               or brightness >= 0 and (isinstance(brightness, float) or isinstance(brightness, int)), \
               "Brightness must be a float or tuple of two floats with non-negative value(s)."
        assert isinstance(contrast, tuple) and contrast[0] >= 0 and contrast[1] >= 0 \
               or contrast >= 0 and (isinstance(contrast, float) or isinstance(contrast, int)), \
               "Contrast must be a float or tuple of two floats with non-negative value(s)."
        assert isinstance(saturation, tuple) and saturation[0] >= 0 and saturation[1] >= 0 \
               or saturation >= 0 and (isinstance(saturation, float) or isinstance(saturation, int)), \
               "Saturation must be a float or tuple of two floats with non-negative value(s)."
        assert isinstance(hue, tuple) and -0.5 <= hue[0] <= 0.5 and -0.5 <= hue[1] <= 0.5 \
               or 0 <= hue <= 0.5 and (isinstance(hue, float) or isinstance(hue, int)), \
               "Hue must be a float in [0, 0.5] or a tuple of two floats in [-0.5, 0.5]."

        self.imgs1 = imgs1
        self.imgs2 = imgs2

        self.affine_transformer = RandomAffine2Img(degrees, translate=translate, scale=scale, fill=fill)
        self.color_transformer = ColorJitter2Img(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __transform_images(self, index):
        """
        Applies required transformations to images at index passed. The same transformation is applied to both images.
        We first apply affine transformations, then color transformations.
        :param index: Index of requested images
        :return: Tuple of augmented images
        """

        img1 = self.imgs1[index]
        img2 = self.imgs2[index]

        # Apply transformations
        img1, img2 = self.affine_transformer(img1, img2)
        img1, img2 = self.color_transformer(img1, img2)

        return img1, img2

    def __getitem__(self, index):
        img1, img2 = self.__transform_images(index)
        return img1, img2

    def __len__(self):
        return len(self.imgs1)
