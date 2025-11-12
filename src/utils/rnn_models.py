import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, rnn_type="LSTM", dropout=0.2):
        super().__init__()
        self.rnn_type = rnn_type.upper()
        rnn_cls = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def prepare_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["ma_7d"] = df["close"].rolling(7).mean()
    df["ma_30d"] = df["close"].rolling(30).mean()
    df["volatility"] = df["return"].rolling(7).std()
    df = df.dropna()
    df = df.set_index("date")
    return df


def make_windowed_multivariate(df_scaled, target_col="close", window=60):
    X, y = [], []
    data = df_scaled.values
    target_idx = df_scaled.columns.get_loc(target_col)
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def invert_scaling(preds_scaled, scaler, df_scaled, target_col="close"):
    dummy = np.zeros((len(preds_scaled), df_scaled.shape[1]))
    dummy[:, df_scaled.columns.get_loc(target_col)] = preds_scaled
    return scaler.inverse_transform(dummy)[:, df_scaled.columns.get_loc(target_col)]


def train_torch_rnn(X, y,
                    rnn_type="LSTM",
                    hidden_size=128,
                    num_layers=3,
                    batch_size=32,
                    epochs=150,
                    lr=5e-4,
                    patience=15):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TorchRNN(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, rnn_type=rnn_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} | Best loss: {best_loss:.6f}")
            break

    return model, {"train_loss": best_loss}


def train_crypto_model(df, coin_name):
    """
    Aplica todo el flujo:
    1. Prepara características
    2. Escala datos
    3. Divide en train/test
    4. Genera ventanas
    5. Entrena modelo
    """
    print(f"\n🟢 Entrenando modelo para {coin_name}...\n")

    # 1️⃣ Preparar datos
    df = prepare_features(df)
    cols = ["close", "log_return", "volatility", "volume", "ma_7d", "ma_30d"]

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[cols]),
        columns=cols,
        index=df.index
    )

    # 2️⃣ Dividir datos
    split_idx = int(len(df_scaled) * 0.8)
    train_scaled, test_scaled = df_scaled.iloc[:split_idx], df_scaled.iloc[split_idx:]
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    # 3️⃣ Ventanas
    X_train, y_train = make_windowed_multivariate(train_scaled, "close", window=60)
    X_test, y_test = make_windowed_multivariate(test_scaled, "close", window=60)

    # 4️⃣ Entrenamiento
    model, hist = train_torch_rnn(
        X_train, y_train,
        rnn_type="LSTM",
        hidden_size=128,
        num_layers=3,
        batch_size=32,
        epochs=150,
        lr=5e-4,
        patience=15
    )

    # 5️⃣ Evaluación
    model.eval()
    with torch.no_grad():
        X_torch = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds_scaled = model(X_torch).cpu().numpy().flatten()

    preds = invert_scaling(preds_scaled, scaler, df_scaled, "close")
    y_real = invert_scaling(y_test, scaler, df_scaled, "close")

    mae = mean_absolute_error(y_real, preds)
    rmse = np.sqrt(mean_squared_error(y_real, preds))
    print(f"📊 {coin_name} | MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # 6️⃣ Gráfica
    plt.figure(figsize=(12, 5))
    plt.plot(train_df.index, train_df["close"], label="Entrenamiento")
    plt.plot(test_df.index, test_df["close"], label="Prueba real")
    plt.plot(test_df.index[60:], preds, label="Predicción", color="green", linestyle="--")
    plt.title(f"Predicción LSTM — {coin_name}")
    plt.legend()
    plt.show()

    return model, scaler, df_scaled
