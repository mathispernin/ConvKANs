import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Trainer:
    """Trainer pour modèles KAN."""

    def __init__(self, model, device='cpu', lr=1e-3, weight_decay=1e-4, reg_lambda=0.0, epochs=30):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.history = {
            'train_loss': [], 
            'train_acc': [],
            'val_loss': [], 
            'val_acc': [],
            'val_f1': [], 
            'val_precision': [], 
            'val_recall': [],
            'epoch_time': []
        }

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Calcul de la perte standard
            loss = self.criterion(outputs, targets)

            # Ajout de la régularisation personnalisée si disponible
            if hasattr(self.model, 'regularization_loss'):
                reg_loss = self.model.regularization_loss()
                loss = loss + self.reg_lambda * reg_loss
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(train_loader), 100. * correct / total

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        return metrics

    def fit(self, train_loader, val_loader, epochs=None, verbose=True):
        if epochs is None:
            epochs = self.epochs

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            epoch_time = time.time() - start_time

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['epoch_time'].append(epoch_time)

            self.scheduler.step()

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Time: {epoch_time:.2f}s | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | "
                      f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                      f"F1: {val_metrics['f1']:.4f}")

        return self.history

    def get_final_metrics(self):
        return {
            'accuracy': self.history['val_acc'][-1],
            'f1': self.history['val_f1'][-1],
            'precision': self.history['val_precision'][-1],
            'recall': self.history['val_recall'][-1],
            'avg_epoch_time': sum(self.history['epoch_time']) / len(self.history['epoch_time'])
        }
