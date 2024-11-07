from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset
import os
import torch
import numpy as np


class MatDataset(Dataset):
    def __init__(self, folder_path, data, label, transform=None):
        self.folder_path = folder_path
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        datas = loadmat(file_path)
        data = datas[self.data]
        data = np.array(data).astype(np.float32)
        data_tensor = torch.from_numpy(data)
        label = datas[self.label]
        label = np.array(label).astype(np.float32)
        label_tensor = torch.from_numpy(label)
        return data_tensor, label_tensor

if __name__ == '__main__':

    val_dataset = MatDataset('./result/val/', 'data', 'label')
    val_loader = DataLoader(val_dataset, batch_size=7, shuffle=True)

    # load trained model
    AE = torch.load('Trained model.pkl', map_location=torch.device('cpu'))

    data, label = next(iter(val_loader))
    data = data.unsqueeze(1)
    label = label.unsqueeze(1)

    AE_latentcode, AE_output = AE(data)

    AE_output_cpu = AE_output.cpu()
    AE_output_np = AE_output_cpu.detach().numpy()
    AE_output_img = AE_output_np.squeeze()

    AE_input = data.cpu()
    AE_input_np = AE_input.detach().numpy()
    AE_input_img = AE_input_np.squeeze()

    # show several samples
    val_idx = [os.path.splitext(file)[0] for file in val_dataset.file_list ]

    for i in range(AE_output_img.shape[0]):
        IMG_output = AE_output_img[i, :, :]
        IMG_output = IMG_output.squeeze()

        plt.subplot(1, 2, 1)
        plt.imshow(IMG_output)
        plt.axis('off')
        plt.title('Output')

        IMG_input = AE_input_img[i, :, :]
        IMG_input = IMG_input.squeeze()

        plt.subplot(1, 2, 2)
        plt.imshow(IMG_input)
        plt.axis('off')
        plt.title('Input')


        image_path = './result/val/' + f'/Test_{val_idx[i]}#sample.png'
        plt.savefig(image_path)

        plt.show()