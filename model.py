import torch
import json
from src.Noise2noise import TestNet
import src.utils as utils 
from datetime import datetime
import time as time
import src.data_iterator as di
from torch.utils.data import DataLoader 

class Model():
    def __init__(self):
        """
        Instantiates the model class.
        :return: None
        """

        # Not very pretty in constructor, but best solution so far. 
        with open("src/parameters.json", "r") as read_file:
            self.params = json.load(read_file)
        
        # Loads the model that we want to train, according to the config file 
        self.model = utils.get_model(self.params)
        # Loads the best model with pre-trained weights
        # TODO 
        self.best_model = None
        # To store the training logs 
        # First row: the epoch number
        # Second row: the training error 
        # Third row: the validation error
        self.logs =  [[],[],[]]

    def load_pretrained_model(self):
        """
        Loads best model from file bestmodel.pth
        :return: None
        """
        self.best_model = torch.load(self.params["path_model"]+self.params["best_model"])

    def train(self, train_input, train_target):
        """
        Trains the model.
        :param train_input: Training data.
        :param train_target: Train targets.
        :return: None
        """
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("You are using the device: "+str(device))

        # Custom train/validation split - Start by shuffling and sending to GPU is available 
        idx = torch.randperm(train_input.size()[0])
        train_input = train_input[idx,:,:,:]
        train_target = train_target[idx,:,:,:]
        # Then take the last images as validation set (w.r.t. proportion)
        split = int(self.params["validation"]*train_input.size(0))
        # Training data is standardized by the DataLoader 
        val_input  = (train_input[split:-1]/255).to(device)
        val_target = (train_target[split:-1]/255).to(device)
        train_input = (train_input[0:split]).to(device)
        train_target = (train_target[0:split]).to(device)
        # Data augmentation and dataloader 
        # Parameters for augmentation
        degrees = 180
        translate = (0.1, 0.1)
        scale = (0.9, 1.2)
        brightness = (0.9, 1)
        contrast = (0.7, 1)
        saturation = (0.5, 1)
        hue = (0, 0.4)
        data_iter = di.DataIterator(train_input, train_target,
            degrees=degrees, translate=translate, scale=scale, 
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        data_loader = DataLoader(data_iter, batch_size=self.params["batch_size"], shuffle=True, num_workers=0)
        
        self.model = self.model.to(device)
        # Set the model in train mode
        self.model.train(True)

        # Initialize optimizer
        optimizer = utils.get_optimizer(self.model, self.params["opti_type"],self.params["lr"])
        # The error function 
        criterion = utils.get_loss(self.params["error"])
        # Maximum number of epochs/iterations 
        n_max = self.params["max_iter"]

        # Monitor time taken
        start = time.time()
        # The loop on the epochs
        for epoch in range(0, n_max):
            for train_img, target_img in data_loader:
                output = self.model(train_img)
                loss = criterion(output, target_img)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Evaluate the model every eval_step
            if (epoch+1)%self.params["eval_step"]==0:
                self.model.train(False)
                with torch.no_grad():
                    train_pred = self.model(train_input/255)
                    val_pred = self.model(val_input)
                    self.logs[0].append(epoch)
                    self.logs[1].append(criterion(train_pred, train_target/255))
                    self.logs[2].append(criterion(val_pred, val_target))
                self.model.train(True)
                utils.waiting_bar(epoch, n_max, (self.logs[1][-1], self.logs[2][-1]))
        
        # Save the model - path name contains the parameters + date
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = self.params["model"]+"_"+self.params["opti_type"] \
                +"_"+str(self.params["error"])+"_"+str(self.params["lr"])+"_"+str(self.params["batch_size"])+"_"+date+".pth"
       
        torch.save(self.model, self.params["path_model"]+path)
        # Save the logs as well
        self.logs =  torch.tensor(self.logs)
        torch.save(self.logs, self.params["path_logs"]+path)

        # Record and print time 
        end = time.time()
        min = (end-start)//60
        sec = (end-start)%60
        print("\nTime taken for training: {:.0f} min {:.0f} s".format(min, sec))



    def predict(self, test_input):
        """
        Predicts with the model on the provided input.
        :param test_input: Test input.
        :return: The prediction (torch.Tensor).
        """
        # Set the model in testing mode
        self.model.train(False)
        return self.model(test_input)

    def psnr(self, denoised, ground_truth):
        """
        Computes the Peak Signal-to-Noise Ratio of a denoised image compared to the ground truth.
        :param denoised: Denoised image. Must be in range [0, 1].
        :param ground_truth: Ground truth image. Must be in range [0, 1].
        :return: PSNR (0-dimensional torch.Tensor)
        """

        assert denoised.shape == ground_truth.shape, "Denoised image and ground truth must have the same shape!"

        mse = torch.mean((denoised - ground_truth) ** 2)
        return -10 * torch.log10(mse + 10 ** -8)
