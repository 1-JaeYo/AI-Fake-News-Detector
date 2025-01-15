from tqdm import tqdm
import torch


def train_model(model, train_loader, optimizer, scheduler, device, num_epochs=3, accumulation_steps=4):
    model.to(device)  # Ensure model is on the CPU
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        loop = tqdm(train_loader, leave=True)  # Progress bar for batches

        for step, batch in enumerate(loop):
            # Forward pass
            inputs = {key: val.to(device) for key, val in batch.items()}  # Move data to CPU
            outputs = model(**inputs)  # Get outputs from model

            # Compute loss
            loss = outputs.loss / accumulation_steps  # Divide by accumulation steps
            loss.backward()  # Backpropagation

            # Perform weight update after accumulation steps
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()  # Accumulate loss

            # Update progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

        # Log average loss for the epoch
        print(f"Epoch {epoch} Average Loss: {total_loss / len(train_loader)}")
