import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_data(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, len(dataset.classes)


def analyze_system(ram_gb, cpu_cores, gpu):
    # Simple rule-based logic (can improve later)
    if ram_gb < 8:
        batch = 8
    elif ram_gb < 16:
        batch = 16
    else:
        batch = 32

    return {
        "recommended_batch": batch
    }