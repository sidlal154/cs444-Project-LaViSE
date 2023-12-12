import matplotlib.pyplot as plt
import os

def plot_loss_accuracy(files):
    plt.figure(figsize=(10, 6))

    for file_path in files:
        epochs = []
        train_loss = []
        valid_loss = []

        label = os.path.basename(file_path).split('_')[1].split('.')[0]

        with open(file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                if line.startswith("epoch"):
                    epoch = int(line.split(":")[1].strip())
                    epochs.append(epoch)
                elif line.startswith("train loss"):
                    train_loss.append(float(line.split(":")[1].strip()))
                elif line.startswith("valid loss"):
                    valid_loss.append(float(line.split(":")[1].strip()))

        plt.plot(epochs, train_loss, label=f'Train Loss ({label})', linestyle='--')
        plt.plot(epochs, valid_loss, label=f'Validation Loss ({label})', linestyle=':')


    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# Example usage with multiple files
file_paths = ['valid_LaViSE.txt','valid_FaLaViSE.txt','valid_BeLaViSE.txt']
plot_loss_accuracy(file_paths)
