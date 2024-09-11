import mlflow
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict

if len(sys.argv) > 1:
    limit = int(sys.argv[1])
else:
    limit = None

mlflow.set_experiment("Fake News Model Training")


class FNDetector(nn.Module):
    def __init__(self):
        super(FNDetector, self).__init__()

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.base_model = RobertaModel.from_pretrained(
            'roberta-base').to(self.device)

        self.max_tokens = self.base_model.config.max_position_embeddings
        self.emb_dim = self.base_model(
            **self.tokenizer(
                "hello roberta",
                return_tensors="pt"
            ).to(self.device)
        )[1].shape[1]

        self.classification_head = nn.Sequential(
            nn.Linear(self.emb_dim, 1),
        )
        self.dropout = nn.Dropout(0.5)
        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask)
        pooler = outputs[1]
        pooler = self.dropout(pooler)
        outputs = self.classification_head(pooler)

        return outputs


model = FNDetector()

df = pd.read_csv("final_combined_dataset.csv", nrows=limit)

x = []
to_remove = []
for idx, i in tqdm(enumerate(df["Text"].values.astype(str)), total=df.shape[0]):
    tokens = model.tokenizer(i, return_tensors="pt",
                             padding="max_length", truncation=True)

    tokens["input_ids"] = torch.squeeze(tokens["input_ids"])
    tokens["attention_mask"] = torch.squeeze(tokens["attention_mask"])

    if tokens["input_ids"].shape[0] > model.max_tokens:
        to_remove.append(idx)
    x.append(tokens)

df["x"] = x
to_remove = list(df.iloc[to_remove, :]["Text"].index)
df = df.drop(labels=to_remove, axis=0)


def split_data(df, train_frac=0.8):
    df = df.sample(frac=1.0)
    train_size = round(df.shape[0]*train_frac)
    train_df = df.iloc[:train_size, :]
    test_df = df.iloc[train_size:, :]
    return train_df, test_df


class FNDataset(Dataset):
    def __init__(self, df):
        self.data = df["x"].values
        self.labels = (df["Label"] == "fake").astype(int).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_model(df):
    batch_size = 2
    lr = 0.001
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train, test = split_data(df)

    train_dataset = FNDataset(train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = FNDataset(test)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 5
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    hyperparams = {
        "epochs": num_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": lr,
    }

    history = defaultdict(float)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in tqdm(
            train_loader,
            total=int(train.shape[0]/batch_size),
            desc=f"Epoch {epoch}"
        ):
            X_batch, y_batch = X_batch.to(
                model.device), y_batch.to(model.device)

            optimizer.zero_grad()
            outputs = model(**X_batch)

            loss = criterion(
                torch.squeeze(outputs.float()),
                torch.squeeze(y_batch.float())
            )
            loss .backward()
            optimizer.step()

        history["loss"] = loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(
                    model.device), y_batch.to(model.device)
                outputs = model(**X_batch)
                validation_loss = criterion(
                    torch.squeeze(outputs.float()),
                    torch.squeeze(y_batch.float())
                )
                val_loss += validation_loss.item()

        val_loss /= len(test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    # Logging only the last (best) loss of this run
    history["loss"] = loss.item()
    history["val_loss"] = validation_loss.item()

    with mlflow.start_run():
        mlflow.log_params(hyperparams)
        mlflow.log_metrics(history)

    return model


train_model(df)
