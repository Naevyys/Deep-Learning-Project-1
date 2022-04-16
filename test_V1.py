import torch
import src.utils 
from model import Model
import matplotlib.pyplot as plt

#Load the images
noisy_imgs_1, noisy_imgs_2 = torch.load('data/train_data.pkl')
noisy_imgs , clean_imgs = torch.load('data/val_data.pkl')
# Create the model
test_model = Model()
print(test_model.model)
# Train the model
# Normalize between 0 and 1 like this because Iris 
# wants to use branches and hasn't uploaded here data transform yet
test_model.train(noisy_imgs_1[0:1000,:,:,:]/256, noisy_imgs_2[0:1000,:,:,:]/256)
print("Successful training!")
# Plot the results 
plt.plot(test_model.logs[0],test_model.logs[1], label="Train loss")
plt.plot(test_model.logs[0],test_model.logs[2], label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel(test_model.params["error"]+" Loss")
plt.title(test_model.params["model"])
plt.legend() 
plt.show()
# Test the model 
predictions = test_model.predict(noisy_imgs/256)
# Test the PNSR 
pnsr = test_model.psnr(predictions, clean_imgs)
print("The psnr is: %f"%pnsr)

# Load and print the best model 
test_model.load_pretrained_model()
print(test_model.best_model) 

