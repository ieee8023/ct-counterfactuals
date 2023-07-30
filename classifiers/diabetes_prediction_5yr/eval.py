import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import monai

sys.path.insert(0, '/dataNAS/people/lblankem/')

from chronic_disease_prediction_3d import dataset

# Initialize the dataset
CSV_FILE = '/dataNAS/people/lblankem/chronic_disease_prediction_3d/data/labels_adjusted_all.csv'

# Initialize the test dataset and loader
test_dataset = dataset.DiabetesPredictionDataset(CSV_FILE, 'test')
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Initialize the model and criterion
model = monai.networks.nets.densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=1)
criterion = nn.BCEWithLogitsLoss()

# Load the model from a file
MODEL_FILE = '/dataNAS/people/lblankem/chronic_disease_prediction_3d/models/diabetes_5yr.pt'
model.load_state_dict(torch.load(MODEL_FILE))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Evaluation loop
model.eval()
outputs = []
labels = []
with torch.no_grad():
    test_loss = 0
    for i, (img_data, dm_label) in enumerate(test_loader):
        img_data = img_data.to(device)
        dm_label = dm_label.to(device)

        output = model(img_data)
        loss = criterion(output, dm_label.unsqueeze(1))
        test_loss += loss.item()

        outputs.extend(output.detach().cpu().numpy())
        labels.extend(dm_label.detach().cpu().numpy())

    outputs = np.array(outputs)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, outputs)
    auprc = average_precision_score(labels, outputs)

    # print the number of scans with diabetes
    print(f'Number of scans with diabetes: {np.sum(labels)}')
    print(f"Number of scans without diabetes: {len(labels) - np.sum(labels)}")
    print(f'Test Loss: {test_loss/len(test_loader)}, AUROC: {auroc}, AUPRC: {auprc}')
