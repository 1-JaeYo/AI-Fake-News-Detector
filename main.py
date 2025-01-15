import torch
from data_processing import load_and_prepare_data, tokenize_data
from dataset import FakeNewsDataset
from model import initialize_model, setup_optimizer_and_scheduler
from train import train_model
from evaluate import evaluate_model
from torch.utils.data import DataLoader

# Load and process data
X_train, X_test, y_train, y_test = load_and_prepare_data("archive/Fake.csv", "archive/True.csv")

# Tokenize
train_encodings = tokenize_data(X_train)
test_encodings = tokenize_data(X_test)

# Create datasets
train_dataset = FakeNewsDataset(train_encodings, y_train.tolist())
test_dataset = FakeNewsDataset(test_encodings, y_test.tolist())

# DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model and training setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = initialize_model()
optimizer, scheduler = setup_optimizer_and_scheduler(model, train_dataset)

# Train the model
train_model(model, train_loader, optimizer, scheduler, device)

# Evaluate the model
evaluate_model(model, test_loader, device)

# Save the model
model.save_pretrained("fake_news_detector")
