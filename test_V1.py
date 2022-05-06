from ctypes import util
import torch
import src.utils 
from model import Model
import matplotlib.pyplot as plt

#Load the images
noisy_imgs_1, noisy_imgs_2 = torch.load('data/train_data.pkl')
# Send to GPU if available 
# noisy_imgs_1, noisy_imgs_2 = src.utils.to_cuda(noisy_imgs_1), src.utils.to_cuda(noisy_imgs_2)
# noisy_imgs, clean_imgs = src.utils.to_cuda(noisy_imgs), src.utils.to_cuda(clean_imgs)
# Create the model
test_model = Model()
print(test_model.model)
# Train the model
# Normalize between 0 and 1 like this because Iris 
# wants to use branches and hasn't uploaded here data transform yet
test_model.train(noisy_imgs_1, noisy_imgs_2)
print("Successful training!")
# Plot the results 
plt.plot(test_model.logs[0].cpu(),test_model.logs[1].cpu(), label="Train loss")
plt.plot(test_model.logs[0].cpu(),test_model.logs[2].cpu(), label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel(test_model.params["error"]+" Loss")
plt.title(test_model.params["model"])
plt.legend() 
plt.show()

# Load the evaluation data
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
noisy_imgs , clean_imgs = torch.load('data/val_data.pkl')
noisy_imgs , clean_imgs = noisy_imgs.to(device) , clean_imgs.to(device)
# Test the model 
predictions = test_model.predict(noisy_imgs/255)
# Test the PNSR 
pnsr = test_model.psnr(predictions, clean_imgs/255)
print("The psnr is: %f"%pnsr)

# Load and print the best model 
test_model.load_pretrained_model()
print(test_model.best_model) 

