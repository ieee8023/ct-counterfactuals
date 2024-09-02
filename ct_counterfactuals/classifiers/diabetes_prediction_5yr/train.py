import sys
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import monai

sys.path.insert(0, '/dataNAS/people/lblankem/')

from chronic_disease_prediction_3d import dataset

# Initialize the dataset
CSV_FILE = '/dataNAS/people/lblankem/chronic_disease_prediction_3d/data/labels_adjusted_all.csv'

train_dataset = dataset.DiabetesPredictionDataset(CSV_FILE, 'train')
val_dataset = dataset.DiabetesPredictionDataset(CSV_FILE, 'val')

# Initialize the model, criterion, and optimizer
model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Set up for model saving
save_dir = '/dataNAS/people/lblankem/chronic_disease_prediction_3d/models/'  # Directory where you want to save the models
best_loss = np.inf  # Initialize the best loss as infinity

# Training loop
print_every = 1
validate_every = 200  # Change this to your preferred validation step
steps = 0
for epoch in range(10):
    model.train()
    train_loss = 0
    for i, (img_data, dm_label) in enumerate(train_loader):
        img_data = img_data.to(device)
        dm_label = dm_label.to(device)

        optimizer.zero_grad()
        output = model(img_data)
        loss = criterion(output, dm_label.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        steps += 1

        if (i + 1) % print_every == 0:
            print(f'Epoch {epoch+1}, Batch {i+1}, Running Train Loss: {train_loss / (i+1)}')
            sys.stdout.flush()

        if steps % validate_every == 0:  # Validate every 'validate_every' steps
            # Validation loop
            model.eval()
            with torch.no_grad():
                val_loss = 0
                outputs = []
                labels = []
                for j, (img_data, dm_label) in enumerate(val_loader):
                    if j == 200:
                        break
                    img_data = img_data.to(device)
                    dm_label = dm_label.to(device)

                    output = model(img_data)
                    loss = criterion(output, dm_label.unsqueeze(1))

                    val_loss += loss.item()

                    outputs.extend(output.detach().cpu().numpy())
                    labels.extend(dm_label.detach().cpu().numpy())

                outputs = np.array(outputs)
                labels = np.array(labels)

                auroc = roc_auc_score(labels, outputs)
                auprc = average_precision_score(labels, outputs)
                
                # print the number of scans with diabetes
                print(f'Number of scans with diabetes: {np.sum(labels)}')
                print(f"Number of scans without diabetes: {len(labels) - np.sum(labels)}")
                print(f'Epoch {epoch+1}, Step {steps}, Val Loss: {val_loss/j}, AUROC: {auroc}, AUPRC: {auprc}')

                if val_loss / j < best_loss:  # If the current model has the best validation loss
                    best_loss = val_loss / j
                    torch.save(model.state_dict(), save_dir + 'best_model_every_low_lr.pt')  # Save the best model

                torch.save(model.state_dict(), save_dir + 'lastest_model_every_low_lr.pt')  # Save the latest model

            model.train()  # Switch back to training mode

    print(f'Epoch {epoch+1}, Average Train Loss: {train_loss / len(train_loader)}')


