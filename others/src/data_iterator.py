import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import affine, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


class RandomRotation2Img():

    def __init__(self, prob):
        """
        :param prob:
        """
        self.prob = prob

    def __call__(self, img1, img2):
        """
        Applies the same random affine transformations to two images
        :param img1: First image
        :param img2: Second image
        :param prob: Probability of transformation
        :return: Tuple of transformed images
        """
        angles = [90, 180, -90]
        if torch.rand(1) < self.prob:
            angle_index = torch.randint(0, 3, (1,))
            transformed_img1 = affine(img1, angle=angles[angle_index], scale=1, shear=[0.], translate=[0, 0])
            transformed_img2 = affine(img2, angle=angles[angle_index], scale=1, shear=[0.], translate=[0, 0])
            return transformed_img1, transformed_img2
        else:
            return img1, img2


class ColorJitter2Img(ColorJitter):

    def __init__(self, prob, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def __call__(self, img1, img2):
        """
        Applies the same random color transformations to two images
        :param img1: First image
        :param img2: Second image
        :return: Tuple of transformed images
        """

        # Get random parameters
        order, brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation,
                                                                       self.hue)

        # Transform images
        transformed_img1 = img1
        transformed_img2 = img2
        for i in order:
            if i == 0 and brightness is not None and torch.rand(1) < self.prob:
                transformed_img1 = adjust_brightness(transformed_img1, brightness)
                transformed_img2 = adjust_brightness(transformed_img2, brightness)
            elif i == 1 and contrast is not None and torch.rand(1) < self.prob:
                transformed_img1 = adjust_contrast(transformed_img1, contrast)
                transformed_img2 = adjust_contrast(transformed_img2, contrast)
            elif i == 2 and saturation is not None and torch.rand(1) < self.prob:
                transformed_img1 = adjust_saturation(transformed_img1, saturation)
                transformed_img2 = adjust_saturation(transformed_img2, saturation)
            elif i == 3 and hue is not None and torch.rand(1) < self.prob:
                transformed_img1 = adjust_hue(transformed_img1, hue)
                transformed_img2 = adjust_hue(transformed_img2, hue)

        return transformed_img1, transformed_img2


class StandardizeImg():
    def __call__(self, img1, img2):
        """
        Standardize the tensor of images between 0 and 1.
        :param imgs: Tensor containing the images
        :return: Tensor containing the standardized images
        """
        assert torch.is_tensor(img1), "Argument must be a torch tensor"
        assert torch.is_tensor(img2), "Argument must be a torch tensor"
        min_val = min(torch.min(img1), torch.min(img2))
        max_val = max(torch.max(img1), torch.max(img2))
        return (img1-min_val) / max_val, (img2-min_val) / max_val


class DataIterator(Dataset):
    def __init__(self, imgs1, imgs2, prob=0.0, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        """
        Initializes a data iterator for our data.
        :param imgs1: Tensor of first images.
        :param imgs2: Tensor of second images.
        :param prob: Probability of transformation.
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
        # - No need to balance our data, since we do not classify anything
        # - This iterator does not avoid loading everything into memory, since we first load the entire data, then
        #   pass it to the interator


        assert isinstance(imgs1, torch.Tensor) and isinstance(imgs2, torch.Tensor), "Images must be tensors!"
        assert imgs1.size() == imgs2.size(), "Tensors of first and second images must have the same size!"
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

        self.rotation_transformer = RandomRotation2Img(prob=prob)
        self.color_transformer = ColorJitter2Img(prob=prob, brightness=brightness, contrast=contrast,
                                                 saturation=saturation, hue=hue)
        self.standardizer = StandardizeImg()

        self.imgs1, self.imgs2 = self.standardizer(imgs1, imgs2)

    def __transform_images(self, index):
        """
        Applies required transformations to images at index passed. The same transformation is applied to both images.
        We first apply affine transformations, then color transformations.
        :param index: Index of requested images
        :return: Tuple of augmented images
        """

        img1 = self.imgs1[index].clone()
        img2 = self.imgs2[index].clone()

        # Apply transformations
        if len(img1) == 1:
            img1, img2 = self.rotation_transformer(img1, img2)
            img1, img2 = self.color_transformer(img1, img2)
        else:
            for n in range(len(img1)):
                img1[n], img2[n] = self.rotation_transformer(img1[n], img2[n])
                img1[n], img2[n] = self.color_transformer(img1[n], img2[n])
        return img1, img2

    def __getitem__(self, index):
        img1, img2 = self.__transform_images(index)
        return img1, img2

    def __len__(self):
        return len(self.imgs1)
