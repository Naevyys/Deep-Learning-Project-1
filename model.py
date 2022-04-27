import torch
import json
from src.Noise2noise import TestNet
import src.utils as utils 
from datetime import datetime

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

        # Custom train/validation split - Start by shuffling
        idx = torch.randperm(train_input.size()[0])
        train_input = train_input[idx,:,:,:]
        train_target = train_target[idx,:,:,:]
        # Then take the last images as validation set (w.r.t. proportion)
        split = int(self.params["validation"]*train_input.size(0))
        # Send everything to GPU, if available otherwise stays on CPU
        val_input  = utils.to_cuda(train_input[split:-1])
        val_target = utils.to_cuda(train_target[split:-1])
        train_input = utils.to_cuda(train_input[0:split])
        train_target = utils.to_cuda(train_target[0:split])
        
        self.model = utils.to_cuda(self.model)
        # Set the model in train mode
        self.model.train(True)

        # Initialize optimizer
        optimizer = utils.get_optimizer(self.model, self.params["opti_type"],self.params["lr"])
        # The error function 
        criterion = utils.get_loss(self.params["error"])
        # Maximum number of epochs/iterations 
        n_max = self.params["max_iter"]

        # The loop on the epochs
        for epoch in range(0, n_max):
            # Shuffle the data set - probably not efficient
            idx = torch.randperm(train_input.size()[0])
            train_input = train_input[idx,:,:,:]
            train_target = train_target[idx,:,:,:]

            mini_batch_size = self.params["batch_size"]
            # The loop for the mini batches
            for b in range(0, train_input.size(0), mini_batch_size):
                # Not a very efficient way to separate the mini batches
                # But does the job
                if b+mini_batch_size<train_input.size(0):
                    output = self.model(train_input.narrow(0, b, mini_batch_size))
                    loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                    self.model.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Evaluate the model every eval_step
            if (epoch+1)%self.params["eval_step"]==0:
                self.model.train(False)
                with torch.no_grad():
                    train_pred = self.model(train_input)
                    val_pred = self.model(val_input)
                    self.logs[0].append(epoch)
                    self.logs[1].append(criterion(train_pred, train_target))
                    self.logs[2].append(criterion(val_pred, val_target))
                self.model.train(True) 
        
        # Save the model - path name contains the parameters + date
        date = datetime.now().strftime("%d%m%Y_%H%M%S")
        path = self.params["model"]+"_"+self.params["opti_type"] \
                +"_"+str(self.params["error"])+"_"+str(self.params["lr"])+"_"+str(self.params["batch_size"])+"_"+date+".pth"
       
        torch.save(self.model, self.params["path_model"]+path)
        # Save the logs as well
        self.logs =  torch.tensor(self.logs)
        torch.save(self.logs, self.params["path_logs"]+path)


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
