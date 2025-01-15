from transformers import DistilBertForSequenceClassification, AdamW, get_scheduler


def initialize_model():
    return DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


def setup_optimizer_and_scheduler(model, train_dataset, num_epochs=3, batch_size=16):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataset) // batch_size * num_epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    return optimizer, scheduler
