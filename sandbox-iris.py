# This is a test file and is not intended to be part
# of the final deliverables.

import torch
from src.data_iterator import DataIterator
import matplotlib.pyplot as plt

# Questions:
# - Can we import os to join paths?
# - Can we import torchvision?
# - Can we use matplotlib for visualisations and plots?

datafolder = "data"
train_data_file = datafolder + "/train_data.pkl"
val_data_file = datafolder + "/val_data.pkl"

# Load training data
noisy_imgs_1, noisy_imgs_2 = torch.load(train_data_file)
# Load validation data
noisy_imgs, clean_imgs = torch.load(val_data_file)

degrees = 180
translate = (0.1, 0.1)
scale = (0.9, 1.2)

iterator = DataIterator(noisy_imgs_1, noisy_imgs_2,
                        degrees=degrees,
                        translate=translate,
                        scale=scale)

random_index = torch.randint(0, 50000, (1,)).item()
sample_img1, sample_img2 = iterator.__getitem__(random_index)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(sample_img1.numpy().transpose((1, 2, 0)))
ax[1].imshow(sample_img2.numpy().transpose((1, 2, 0)))
plt.show()
