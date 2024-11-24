import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# Set the dataset path based on your folder structure
dataset_path = './1-mnist_learning/data'
processed_path = os.path.join(dataset_path, 'MNIST', 'processed', 'training.pt')
is_download = not os.path.exists(processed_path)

# device to use, try to use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load the training and testing datasets
train_dataset = datasets.MNIST(root=dataset_path, train=True, download=is_download, transform=transform)
test_dataset = datasets.MNIST(root=dataset_path, train=False, download=is_download, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test the DataLoader by fetching one batch
images, labels = next(iter(train_loader))
print(f"Batch size (images): {images.shape}")  # [64, 1, 28, 28]
print(f"Batch size (labels): {labels.shape}")  # [64]



# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input layer (28x28 = 784) to first hidden layer (128 neurons)
        self.fc1 = nn.Linear(28 * 28, 512)
        # Hidden layer (128 neurons) to second hidden layer (64 neurons)
        self.fc2 = nn.Linear(512, 256)
        # Output layer (64 neurons to 10 classes for digits 0-9)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten the input (batch size, 1, 28, 28) -> (batch size, 784)
        x = x.view(-1, 28 * 28)

        # Apply ReLU activation to the first and second hidden layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        # Output logits (raw scores for each class)
        x = self.fc5(x)
        return x





# Initialize the model
model = SimpleNN().to(device)
print(model)


# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer (e.g., Adam optimizer)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

# Print confirmation
print("Loss function and optimizer are set.")




# Training loop
num_epochs = 50  # Number of times to go through the entire dataset

# Tracking metrics
train_losses = []


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    scheduler.step()
    running_loss = 0.0  # To track the loss for the epoch
    
    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)
        # Forward pass: Get predictions
        outputs = model(images)
        loss = criterion(outputs, labels)  # Compute loss

        # Backward pass: Zero gradients, compute gradients, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for this batch
        running_loss += loss.item()

    # Append average loss for this epoch
    train_losses.append(running_loss / len(train_loader))    
    # Print epoch statistics
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}")





# Plot the training loss
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.ylim(0,1)
plt.legend()
plt.show()



# Evaluate the model on the test dataset and print accuracy
model.eval()
correct = 0
total = 0
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate probabilities and top 3 predictions
        probs = F.softmax(outputs, dim=1)
        top3_probs, top3_preds = torch.topk(probs, 3, dim=1)

        # Store misclassified examples
        for i in range(len(labels)):
            if predicted[i] != labels[i]:  # Only collect misclassified examples
                misclassified.append((
                    images[i].cpu(),               # Image
                    labels[i].item(),              # True label
                    top3_preds[i].cpu().tolist(),  # Top 3 predicted classes
                    top3_probs[i].cpu().tolist()   # Top 3 probabilities
                ))

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



# Plot misclassified images with top 3 predictions and probabilities
plt.figure(figsize=(12, 6))
for idx, (image, true, top3_preds, top3_probs) in enumerate(misclassified[:10]):
    plt.subplot(2, 5, idx + 1)
    plt.imshow(image.view(28, 28), cmap='gray')
    plt.title(f"True: {true}\n"
              f"1: {top3_preds[0]} ({top3_probs[0]:.2f})\n"
              f"2: {top3_preds[1]} ({top3_probs[1]:.2f})\n"
              f"3: {top3_preds[2]} ({top3_probs[2]:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()
