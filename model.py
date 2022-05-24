import torch
import json
from datetime import datetime
import time as time
from .src import utils
from .src import data_iterator as di
import pathlib



class Model():
    def __init__(self):
        """
        Instantiates the model class.
        :return: None
        """
        # Due to miniproject 2, we need to correct these parameters
        torch.set_default_dtype(torch.float32)
        torch.set_grad_enabled(True)
        self.path = str(pathlib.Path(__file__).parent.resolve())
        # Not very pretty in constructor, but best solution so far.
        # with open("Proj_287452_337635_288228/Miniproject_1/src/parameters.json", "r") as read_file:
        with open(self.path + "/src/parameters.json", "r") as read_file:
            self.params = json.load(read_file)

        # Loads the model that we want to train, according to the config file
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("You are using the device: " + str(self.device))
        self.model = utils.get_model(self.params).to(self.device)
        # To store the training logs
        # First row: the epoch number
        # Second row: the training error
        # Third row: the validation error
        self.logs = [[], [], []]

    def load_pretrained_model(self):
        """
        Loads best model from file bestmodel.pth
        :return: None
        """
        # Make sure that the model can be loaded whether it was trained on CPU or GPU 
        self.model.load_state_dict(
            torch.load(self.path + self.params["best_model"], map_location=lambda storage, loc: storage))
        self.model.eval() 

    def train(self, train_input, train_target, num_epochs=None):
        """
        Trains the model.
        :param train_input: Training data.
        :param train_target: Train targets.
        :param num_epochs: number of epochs
        :return: None
        """

        # Monitor time taken
        start = time.time()

        # Custom train/validation split - Start by shuffling and sending to GPU is available
        idx = torch.randperm(train_input.size()[0])

        train_input = train_input[idx, :, :, :]
        train_target = train_target[idx, :, :, :]
        # Then take the last images as validation set (w.r.t. proportion)
        split = int(self.params["validation"] * train_input.size(0))
        # Training data is standardized by the DataLoader
        val_input = (train_input[0:split] / 255)
        val_target = (train_target[0:split] / 255)
        train_input = (train_input[split:-1])
        train_target = (train_target[split:-1])

        # Data augmentation

        # Parameters for augmentation
        probability = 0.5
        brightness = (0.9, 1)
        contrast = (0.7, 1)
        saturation = (0.5, 1)
        hue = (0, 0.4)

        data_iter = di.DataIterator(train_input, train_target, prob=probability,
                                    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

        train_input_augmented, train_target_augmented = data_iter[:]

        nb_images_train = len(data_iter)
        nb_images_val = len(val_input)

        # Set the model in train mode
        self.model.train(True)

        # Initialize optimizer
        optimizer = utils.get_optimizer(self.model, self.params["opti_type"], self.params["lr"])

        # The error function
        criterion = utils.get_loss(self.params["error"])
        # Maximum number of epochs/iterations
        if num_epochs is None:
            num_epochs = self.params["max_epochs"]

        # The loop on the epochs
        for epoch in range(0, num_epochs):
            idx = torch.randperm(nb_images_train)
            train_input_augmented, train_target_augmented = train_input_augmented[idx], train_target_augmented[idx]
            for train_img, target_img in zip(torch.split(train_input_augmented, self.params["batch_size"]),
                                             torch.split(train_target_augmented, self.params["batch_size"])):
                train_img, target_img = train_img.to(self.device), target_img.to(self.device)
                output = self.model(train_img)
                loss = criterion(output, target_img)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate the model every eval_step
            if (epoch + 1) % self.params["eval_step"] == 0:
                self.model.train(False)
                with torch.no_grad():
                    eva_batch_size = 1000
                    train_error = 0.
                    val_error = 0.
                    # Computing the number of split to compute the mean of the error of each batch
                    if nb_images_train % eva_batch_size == 0:
                        nb_split_train = nb_images_train // eva_batch_size
                    else:
                        nb_split_train = nb_images_train // eva_batch_size + 1

                    if nb_images_val % eva_batch_size == 0:
                        nb_split_val = nb_images_val // eva_batch_size
                    else:
                        nb_split_val = nb_images_val // eva_batch_size + 1

                    train_zip = zip(torch.split(train_input_augmented, eva_batch_size),
                                    torch.split(train_target_augmented, eva_batch_size))
                    val_zip = zip(torch.split(val_input, eva_batch_size), torch.split(val_target, eva_batch_size))

                    for train_img, target_img in train_zip:
                        train_img = train_img.to(self.device)
                        target_img = target_img.to(self.device)
                        train_error += criterion(self.model(train_img), target_img)

                    for val_img, val_img_target in val_zip:
                        val_img = val_img.to(self.device)
                        val_img_target = val_img_target.to(self.device)
                        val_error += criterion(self.model(val_img), val_img_target)

                    train_error = train_error / nb_split_train
                    val_error = val_error / nb_split_val

                    self.logs[0].append(epoch)
                    self.logs[1].append(train_error)
                    self.logs[2].append(val_error)

                self.model.train(True)
                utils.waiting_bar(epoch, num_epochs, (self.logs[1][-1], self.logs[2][-1]))

        # Save the model - path name contains the parameters + date
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = self.params["model"] + "_" + self.params["opti_type"] \
               + "_" + str(self.params["error"]) + "_" + str(self.params["lr"]) + "_" + str(
            self.params["batch_size"]) + "_" + date + ".pth"

        torch.save(self.model.state_dict(), self.path + self.params["path_model"] + path)
        # Save the logs as well
        self.logs = torch.tensor(self.logs)
        torch.save(self.logs, self.path + self.params["path_logs"] + path)

        # Record and print time
        end = time.time()
        minutes = (end - start) // 60
        sec = (end - start) % 60
        print("\nTime taken for training: {:.0f} min {:.0f} s".format(minutes, sec))
        del train_input_augmented, train_target_augmented, train_input, train_target

    def predict(self, test_input):
        """
        Predicts with the model on the provided input.
        :param test_input: Test input.
        :return: The prediction (torch.Tensor).
        """

        test_input = test_input.to(self.device).float()
        self.model.eval()
        out = self.model(test_input / 255.0).float()
        return out * 255

    def psnr(self, denoised, ground_truth):
        """
        Computes the Peak Signal-to-Noise Ratio of a denoised image compared to the ground truth.
        :param denoised: Denoised image. Must be in range [0, 1].
        :param ground_truth: Ground truth image. Must be in range [0, 1].
        :return: PSNR (0-dimensional torch.Tensor)
        """

        assert denoised.shape == ground_truth.shape, "Denoised image and ground truth must have the same shape!"
        return - 10 * torch.log10(((denoised-ground_truth) ** 2).mean((1, 2, 3))).mean()
