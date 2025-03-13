import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef
import pandas as pd
from datasets import MultimodalMPI3DRealComplex


# Hyperparameters
learning_rate = 1e-4
batch_size = 1024
train_epochs = 30
checkpoint_epochs = 10
datapath = "./data/MPI3d_real_complex"
save_dir = "./models/MPI3d_supervise"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

bias_types = ["selection", "perturbation"]


# Model Definition
class SemanticClassifier(nn.Module):
    def __init__(self):
        super(SemanticClassifier, self).__init__()
        self.encoder = models.resnet18(pretrained=False)
        num_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()  # Remove the original classification layer

        self.classifier_heads = nn.ModuleDict({
            "OBJ_COLOR": nn.Linear(num_features, 4),
            "OBJ_SHAPE": nn.Linear(num_features, 4),
            "OBJ_SIZE": nn.Linear(num_features, 2),
            "CAMERA": nn.Linear(num_features, 3),
            "BACKGROUND": nn.Linear(num_features, 3),
            "H_AXIS": nn.Linear(num_features, 40),
            "V_AXIS": nn.Linear(num_features, 40),
        })

    def forward(self, x, semantic_key):
        features = self.encoder(x)
        logits = self.classifier_heads[semantic_key](features)
        return logits


# Initialize datasets
mean_per_channel = [0.00888889, 0.00888889, 0.00830382]  # values from MPI3D-Real-Complex
std_per_channel = [0.08381344, 0.07622504, 0.06356431]   # values from MPI3D-Real-Complex
transform = transforms.Compose([
    transforms.Normalize(mean_per_channel, std_per_channel)])

train_dataset = MultimodalMPI3DRealComplex(datapath, mode="train", transform=transform)
vocab_filepath = train_dataset.vocab_filepath

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MultimodalMPI3DRealComplex(datapath, mode="test", vocab_filepath=vocab_filepath, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Training function
def train_one_epoch(model, loader, semantic_key, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["semantics"][semantic_key].to(device)

        optimizer.zero_grad()
        outputs = model(images, semantic_key)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# Evaluation function
def evaluate(model, loader, semantic_key, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["semantics"][semantic_key].to(device)

            outputs = model(images, semantic_key)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    mcc = matthews_corrcoef(all_labels, all_preds)
    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    
    return accuracy, mcc


# Training & Evaluation Loop
all_semantics = ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND", "H_AXIS", "V_AXIS"]
results = {}

for semantic_key in all_semantics:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(train_epochs):
        loss, acc = train_one_epoch(model, train_loader, semantic_key, criterion, optimizer, device)
        print(f"[{semantic_key}] Epoch [{epoch+1}/{train_epochs}], Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        if (epoch + 1) % checkpoint_epochs == 0 or epoch == train_epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f"{semantic_key}.pth"))

    # Evaluate
    accuracy, mcc = evaluate(model, test_loader, semantic_key, device)

    print(f"[{semantic_key}] Test Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}")
    results[semantic_key] = {"accuracy": accuracy, "mcc": mcc}


# Save results to CSV
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv(os.path.join(save_dir, "supervised_results.csv"))

print("Training and evaluation completed. Results saved to supervised_results.csv")
