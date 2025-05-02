import io
import os
import unicodedata
import string
import glob
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, line_tensor, category_tensor):
        hidden = self.model.init_hidden()
        cell_state= self.model.init_cell_state()
        for i in range(line_tensor.size()[0]):
            output, hidden,cell_state = self.model(line_tensor[i], hidden,cell_state)
        loss = self.criterion(output, category_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def predict(self, input_line):
        print(f"\n> {input_line}")
        with torch.no_grad():
            line_tensor = self.dataloader.line_to_tensor(input_line)
            hidden = self.model.init_hidden()
            cell_state= self.model.init_cell_state()
            for i in range(line_tensor.size()[0]):
                output, hidden,cell_state = self.model(line_tensor[i], hidden,cell_state)
            guess = self.dataloader.category_from_output(output)
            print(guess)

    def train_model(self, n_iters=100000, plot_steps=1000, print_steps=5000, plot=True):
        current_loss = 0
        all_losses = []

        for i in range(n_iters):
            category, line, category_tensor, line_tensor = self.dataloader.random_training_example()
            output, loss = self.train(line_tensor, category_tensor)
            current_loss += loss

            if (i+1) % plot_steps == 0:
                var=current_loss / plot_steps
                var=round(var,4)
                all_losses.append(var)
                current_loss = 0

            if (i+1) % print_steps == 0:
                guess = self.dataloader.category_from_output(output)
                correct = "CORRECT" if guess == category else f"WRONG ({category})"
                print(f"{i+1} ({(i+1)/n_iters*100:.2f}%) {loss:.4f} / {line} → {guess} ({correct})")

        if plot==True:
            print(all_losses)
#             plt.figure()
#             plt.plot(all_losses)
#             plt.xlabel('Iterations')
#             plt.ylabel('Loss')
#             plt.title('Training Loss over Time')
#             plt.show()

        return all_losses  


