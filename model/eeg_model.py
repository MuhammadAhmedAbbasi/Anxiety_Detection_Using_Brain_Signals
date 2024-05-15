import glob
import numpy as np
import os
import torch
import torch.nn as nn

current_path = os.path.dirname(os.path.abspath(__file__))

class eeg_anxiety_model(torch.nn.Module):
    def __init__(self):
        super(eeg_anxiety_model, self).__init__()
        self.fc1 = nn.Linear(900, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer for binary classification
        # Get the paths of all .pth files in the folder
        pth_files = glob.glob(os.path.join(current_path, "*.pth"))
        pth_files.sort(key=lambda x: os.path.basename(x))
        state_dict = torch.load(pth_files[-1]) #加载state_dict
        self.load_state_dict(state_dict)   # 使用加载的state_dict

    # Given fp1 and fp2 data, the model predicts
    # Return value: If true, it means anxiety, otherwise it means normal person
    def predict(self, fp1_fft, fp2_fft) -> bool:
        fp1_fft = np.abs(np.fft.fft(fp1_fft))
        fp1_fft = fp1_fft / np.sum(fp1_fft)
        fp2_fft = np.abs(np.fft.fft(fp2_fft))
        fp2_fft = fp2_fft / np.sum(fp2_fft)
        x = np.concatenate([fp1_fft, fp2_fft], axis = 0)
        x = torch.tensor(x).float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary classification
        return x >= 0.5