import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report


class CNN(nn.Module):
    def __init__(self, filters=(32, 64, 128), dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters[2] * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class AgeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AgeCNN:
    def __init__(self, epochs=10, lr=0.001, filters=(32, 64, 128), dropout=0.5, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CNN(filters=filters, dropout=dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _make_loader(self, X, y, shuffle=False):
        return DataLoader(AgeDataset(X, y), batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train):
        loader = self._make_loader(X_train, y_train, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimiser.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")

    def predict(self, X, y):
        loader = self._make_loader(X, y)
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                preds = torch.argmax(self.model(X_batch), dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        return all_preds, all_labels

    def evaluate(self, X, y):
        y_pred, y_true = self.predict(X, y)
        acc = accuracy_score(y_true, y_pred)

        print("Accuracy:", acc)
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred))

        return acc
    
    def predict_image(self, img):
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
        return torch.argmax(output, dim=1).item()