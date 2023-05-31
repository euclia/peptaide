from jaqpotpy.models import MolecularTorchGeometric, AttentiveFP
from jaqpotpy.datasets import TorchGraphDataset
from rdkit import Chem
from jaqpotpy.models.evaluator import Evaluator
from jaqpotpy.descriptors.molecular import AttentiveFPFeaturizer, PagtnMolGraphFeaturizer
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
import pandas as pd
import torch
import numpy as np

df = pd.read_csv('activity_MIC_thresh10_microg-per-ml.csv')
df.head()

from sklearn.model_selection import train_test_split

smiles = df['SMILES']
activity = df['Activity']

X_train, X_val, y_train, y_val = train_test_split(smiles, activity, test_size=0.20, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=42)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

len(X_train), len(X_val), len(X_test)

featurizer = AttentiveFPFeaturizer()
# featurizer = PagtnMolGraphFeaturizer()

train_dataset = TorchGraphDataset(smiles=X_train, y=y_train, 
                                  task='classification', featurizer=featurizer)

val_dataset = TorchGraphDataset(smiles=X_val, y=y_val, 
                                  task='classification', featurizer=featurizer)

test_dataset = TorchGraphDataset(smiles=X_test, y=y_test, 
                                  task='classification', featurizer=featurizer)

train_dataset.create()
val_dataset.create()

model = AttentiveFP(
    in_channels=39, 
    hidden_channels=80, 
    out_channels=2, 
    edge_dim=10, 
    num_layers=6,
    num_timesteps=3
).jittable()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

val = Evaluator()
val.dataset = val_dataset
val.register_scoring_function('Accuracy', accuracy_score)
val.register_scoring_function('ROCAUC', roc_auc_score)

m = MolecularTorchGeometric(
    dataset=train_dataset, 
    model_nn=model, 
    eval=val, 
    train_batch=262, 
    test_batch=200, 
    epochs=40, 
    optimizer=optimizer, 
    criterion=criterion, 
    device="cpu"
)

m.fit()

m.eval()

