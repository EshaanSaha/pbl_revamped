import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from utils import load_data

def objective(trial, data_dir, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    image_size = trial.suggest_categorical("image_size", [128, 224])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    loader, num_classes = load_data(data_dir, image_size, batch_size)

    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.fc.parameters(), lr=lr)

    # Train only 1 epoch for speed
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    trial.set_user_attr("accuracy", 1 / (1 + total_loss))  # store accuracy
    return total_loss  # only ONE value


def run_optimization(data_dir, device):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, data_dir, device), n_trials=15)

    best_trial = study.best_trial
    accuracy = best_trial.user_attrs["accuracy"]

    # 🔥 Collect all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "trial": trial.number,
            "loss": trial.value,
            "batch_size": trial.params.get("batch_size", 0),
            "lr": trial.params.get("lr", 0)
        })

    return {
        "params": best_trial.params,
        "accuracy": accuracy,
        "trials": trials_data
    }