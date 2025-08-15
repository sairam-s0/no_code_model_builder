# backend/optizer.py
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataset, epochs=5, lr=0.001, optimizer_type="adam", fine_tune=True, prune=False, quantize=False):
    print(f"[Backend] Starting training for {epochs} epochs, LR={lr}, optimizer={optimizer_type}")
    
    # Example dataset unpacking
    train_loader, val_loader = dataset  

    # Optimizer selection
    if optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # If fine-tuning, freeze some layers
    if fine_tune:
        for param in list(model.parameters())[:-2]:  # freeze all but last 2 layers
            param.requires_grad = False

    # Training loop
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"[Backend] Epoch {epoch+1}/{epochs} completed")

    # Optimization steps
    if prune:
        print("[Backend] Applying pruning...")
        from torch.nn.utils import prune as prune_utils
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune_utils.l1_unstructured(module, name="weight", amount=0.4)

    if quantize:
        print("[Backend] Applying quantization...")
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    # Save model
    torch.save(model.state_dict(), "trained_model.pth")
    print("[Backend] Training & optimization complete!")
