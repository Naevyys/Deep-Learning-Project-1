# This is a test file and is not intended to be part
# of the final deliverables.

import torch
from src.data_iterator import DataIterator
import matplotlib.pyplot as plt

# Questions:
# - Can we import os to join paths?
# - Can we import torchvision?
# - Can we use matplotlib for visualisations and plots?


# TODO
# - Normalization for training & standardization of output
# - Decide on the code architecture
#   - Model classes/subclasses
#   - Training implementation
#     - Check how to use GPU & fallback on CPU if GPU unavailable during training
#     - Save training history for plots later!!
#     - Gitignore folders with training history? Not sure...
#   - Saving the model to file (a few lines I think)
#   - Loading the model (1 line I think)
# - Decide on and implement a baseline model
# - Implement training
# - Implement grid/random search & k-fold cross-validation
# - Train and optimize baseline model, here are some ideas:
#   - learning rate
#   - Optimizer
#   - Layers (number of layers, number of units, kernel size, stride, ...)
#   - Activation functions
#   - Loss function
# - Implement more model architectures & optimize params as well

datafolder = "data"
train_data_file = datafolder + "/train_data.pkl"
val_data_file = datafolder + "/val_data.pkl"

# Load training data
noisy_imgs_1, noisy_imgs_2 = torch.load(train_data_file)
# Load validation data
noisy_imgs, clean_imgs = torch.load(val_data_file)

# Parameters for augmentation
degrees = 180
translate = (0.1, 0.1)
scale = (0.9, 1.2)
brightness = (0.9, 1)
contrast = (0.7, 1)
saturation = (0.5, 1)
hue = (0, 0.4)

iterator = DataIterator(noisy_imgs_1, noisy_imgs_2)  # No augmentation
iterator_augmented = DataIterator(noisy_imgs_1, noisy_imgs_2,
    degrees=degrees, translate=translate, scale=scale,  # With affine transformation augmentation
    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)  # With color transformation augmentation

samples = 5
random_index = torch.randint(0, 50000, (samples,))
for index in random_index:
    sample_img1, sample_img2 = iterator.__getitem__(index.item())
    sample_img1_aug, sample_img2_aug = iterator_augmented.__getitem__(index.item())

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(sample_img1.numpy().transpose((1, 2, 0)))  # Original images
    ax[0, 1].imshow(sample_img2.numpy().transpose((1, 2, 0)))
    ax[1, 0].imshow(sample_img1_aug.numpy().transpose((1, 2, 0)))  # Augmented images
    ax[1, 1].imshow(sample_img2_aug.numpy().transpose((1, 2, 0)))
    plt.show()
