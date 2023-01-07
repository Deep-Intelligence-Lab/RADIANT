from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loader_syn import syn
from modified_model import SignalTransformer
import matplotlib.pyplot as plt
import time
import numpy as np
import random
import time
import torch
import torch.nn.functional as F
import torch.nn as nn

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_loader, loss_fn, optimizer, scheduler, epochs=4):
    """
    Train the VIT model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} |")
        print("-"*70)

        # Measure the elapsed time of each epoch
        # t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, (data, target) in enumerate(train_loader):
            batch_counts +=1
            
            # Load batch to GPU
            data, target = data.to(device)[:,:,0:12,0:80].to(torch.float32), target.to(device)[:,0:1024].to(torch.float32)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            pred = model(data)
            logits, attn = pred[0][:], pred[1]

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, target)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()


            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 10 batches
            if (step % 50 == 0 and step != 0) or (step == 9999):
                # Calculate time elapsed for 10 batches
                # time_elapsed = time.time() - t0_batch

                # Print training results
                # print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} |")
                
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} |")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                # t0_batch = time.time()
                
                # Save the model state
                torch.save(model.state_dict(),'weights/pretrain_model_weights.pth')
                torch.save({"avgloss":avg_train_loss}, "losses/pretrain_loss"+str(epoch_i)+".pth")


        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / 10000
        
        print("-"*70)

        print("\n")
    
    print("Training complete!")


if __name__ == "__main__":
    # Set seed.
    set_seed()

    # Check for CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    dataset = syn("syn.npy")
    train_loader = DataLoader(dataset, shuffle=True, batch_size=32)

    # Model Configuration
    config = {
            "img_size": [48,64],
            "in_chans": 1,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
            "n_classes": 1024
    }

    # Initialize model
    model = SignalTransformer(config)
    model.load_state_dict(torch.load("weights_12_1024_bvp.pth"))
    model = model.to(device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(),
                            lr = 3e-4,
                            eps = 1e-6    # Default epsilon value
                        )
    
    # Total epochs
    epochs=4
    
    # Total number of training steps
    total_steps = 1000 * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)


    # Specify loss function
    loss_fn = nn.MSELoss()

    # Start training
    train(model, train_loader, loss_fn, optimizer, scheduler,epochs=24)
    
                          