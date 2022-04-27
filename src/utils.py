import torch
import src.Noise2noise

def to_cuda(x): 
    """
    Checks whether a GPU is present, and if yes, sends x on the GPU
    :param x: A torch element that can be sent on GPU
    :return : A transfered version of x on GPU, if GPU is available
    """

    if torch.cuda.is_available(): 
        # Check whether x is a list or tuple
        if type(x) == tuple or type(x) == list:
            return [y.cuda() for y in x]
        return x.cuda()
    return x

def get_optimizer(model, type="SGD", lr=None):
    """
    Load the optimizer used for training. 
    :param model: A pytorch nn.Module
    :param type: A string, containing the type of optimizer
    :param lr: A double, containing the learning rate
    :return : A torch.optim method 
    """
    if type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif type=="ADA":
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception("Sorry, we could not find the optimizer.") 

# An interesting thing to develop would be a learning rate scheduler
# Such that the lr decreases from 1 to 1e-2, 1e-4 etc. at certain epochs

def get_loss(type="L2"):
    """
    Load the error/loss function. 
    :param type: A string, the error type
    :return : An error function that takes a prediction and target as input
    """
    # Note according to the Noise2noise paper, the L2 and L1 norm are the simplest
    # loss functions, see if we can do better. 

    if type == "L2":
        return torch.nn.MSELoss()
    elif type == "L1": 
        return torch.nn.L1Loss()
    raise Exception("Sorry, we could not find the error function.") 

def get_model(params):
    
    """
    Load the model to train. 
    :param params: A dictionary, the config file
    :return : A torch.nn.Module model
    """
    type = params["model"]
    
    if type == "TestNet":
        return src.Noise2noise.TestNet(params)
    elif type == "Noise2noise": 
        return src.Noise2noise.Noise2noise(params)
    else:
         raise Exception("Sorry, we could not find any model corresponding to: "+params["model"])

def get_logs(path="../outputs/logs/TestNet_SGD_L2_0.01_64_16042022_023735.pth"):
    """
    Returns the logs of a given training.
    :param path: A string, the path of the log 
    :return : A torch.tensor 1st row the epochs, 
             2nd row the training loss, 3r row validation loss
    """ 
    return torch.load(path) 

def waiting_bar(i, length, loss):
    """
        Simple function that prints a progress/waiting bar + the loss
        :param i: Integer, the current element we are working on
        :param length: Integer, the total number of elements we need to work on
        :param loss: Float, The training loss of the system
        :return: Nothing, just print
    """
    left = int(30 * i / length)
    right = 30 - left
    tags = "=" * left
    spaces = " " * right
    print(f"\r[{tags}>{spaces}] Train loss: {loss[0]:.5f} Val loss: {loss[1]:.5f}", sep="", end="", flush=True)
