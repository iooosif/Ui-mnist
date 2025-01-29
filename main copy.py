import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set the device to GPU if available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
#batch_size: number of samples in one data batch.
#learning_rate: learning rate values for each optimizer.
#num_epochs: number of training epochs.
batch_size = 128
# Learning rates for SGD, SGD with momentum, and Adam optimizers
learning_rate = [0.01,0.01,0.0005]
num_epochs = 10

# Data transformation pipeline
#The data is normalized (brought to the range[0,1]).
#The training and test samples from the MNIST dataset are loaded.
#Data loaders are created for batch processing.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
# Load MNIST dataset for training and testing
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
#  Data loaders for batch processing
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Function to define the model
#The neural network is built from:
#Input layer, transforming 28x28 images into a vector of length 784.
#Two hidden layers with 256 and 128 neurons, using ReLU as an activation function.
#Output layer with 10 neurons (according to the number of classes).
def create_model():
    return nn.Sequential(
        # Flatten the 28x28 input images into a vector of size 784
        nn.Flatten(),
        # Fully connected layer with 256 neurons
        nn.Linear(28 * 28, 256),
        # Apply ReLU activation function
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

# Function to train the model
#The model is trained on the training set using backpropagation.
#The average error and accuracy are calculated for each
def train_model(model, optimizer, criterion, train_loader):
    # Set the model to training mode
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    # Iterate over batches of data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backpropagation to compute gradients
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)# Get predicted class with highest score
        correct += (preds == labels).sum().item()
        total += labels.size(0)# Update total number of samples
    return total_loss / len(train_loader), correct / total

# Function to evaluate the model
#The model is evaluated on the test set.
#Results include mean error, accuracy, predictions, and true labels.
def evaluate_model(model, criterion, test_loader):
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(test_loader), correct / total, all_preds, all_labels

# Optimizer functions
optimizers = {
    "SGD": lambda params: optim.SGD(params, lr=learning_rate[0]),
    "SGD with Momentum": lambda params: optim.SGD(params, lr=learning_rate[1], momentum=0.9),
    "Adam": lambda params: optim.Adam(params, lr=learning_rate[2])
}

# Loss function
# Cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Training and evaluation loop for each optimizer
results = {}
for name, opt_func in optimizers.items():
    print(f"Training with {name}")
    model = create_model()
    optimizer = opt_func(model.parameters())

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, optimizer, criterion, train_loader)
        test_loss, test_acc, _, _ = evaluate_model(model, criterion, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save results for this optimizer
    results[name] = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "model": model
    }

# Plotting test accuracies for all optimizers
for name, result in results.items():
    plt.plot(range(1, num_epochs + 1), result["test_accuracies"], label=f"{name} Test Accuracy")
plt.title("Test Accuracy for Different Optimizers")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate and visualize the best model
best_model_name = max(results, key=lambda x: max(results[x]["test_accuracies"]))
best_model = results[best_model_name]["model"]
_, _, preds, labels = evaluate_model(best_model, criterion, test_loader)

# Plot confusion matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for {best_model_name}")
plt.show()
