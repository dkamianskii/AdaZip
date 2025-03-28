import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from enum import Enum


def set_random_seed(seed):
    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For current GPU
        torch.cuda.manual_seed_all(seed)  # For all GPUs

    # Ensure deterministic behavior (optional but recommended for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the random seed
set_random_seed(42)

class ActivationType(Enum):
    LEAKY_RELU=1
    ELU=2
    GELU=3


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']  # Return the learning rate from the first parameter group


class SequenceDataset(Dataset):
    def __init__(self, data, seq_length, batch_size, in_memory=True):
        self.seq_length = seq_length
        self.batch_size = batch_size
        N = len(data)
        T = N // batch_size
        self.data = data[:batch_size * T].reshape(batch_size, T)
        self.num_batches = (T - 1) // seq_length

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        x = self.data[:, start:end]
        y = self.data[:, start + 1:end + 1]
        return torch.tensor(x, dtype=torch.int32), torch.tensor(y, dtype=torch.long)


class BasePredictor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_tokens, activation_type: ActivationType):
        super(BasePredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.skip_proj = nn.Linear(input_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_tokens)
        match activation_type:
            case ActivationType.LEAKY_RELU:
                self.activation = nn.LeakyReLU()
            case ActivationType.ELU:
                self.activation = nn.ELU()
            case ActivationType.GELU:
                self.activation = nn.GELU()

    def forward(self, x, hidden=None):
        gru_out, hidden_new = self.gru(x, hidden)
        gru_out = self.activation(gru_out)
        skip = self.skip_proj(x)
        out = gru_out + skip
        out = self.layer_norm(out)
        logits = self.fc(out)
        return logits, out, hidden_new


class FirstModel(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_size, activation_type: ActivationType):
        super(FirstModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.predictor = BasePredictor(input_dim=embedding_dim,
                                       hidden_size=hidden_size,
                                       num_tokens=num_tokens,
                                       activation_type=activation_type)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        logits, gru_out, hidden_new = self.predictor(embed, hidden)
        return logits, gru_out, hidden_new, embed


class BoosterModel(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_size, activation_type: ActivationType):
        super(BoosterModel, self).__init__()
        self.predictor = BasePredictor(input_dim=embedding_dim + hidden_size,
                                       hidden_size=hidden_size,
                                       num_tokens=num_tokens,
                                       activation_type=activation_type)

    def forward(self, embed, prev_feature, hidden=None):
        x = torch.cat([embed, prev_feature], dim=-1)
        logits, gru_out, hidden_new = self.predictor(x, hidden)
        return logits, gru_out, hidden_new


def initialize_hidden(batch_size, hidden_size, device):
    return torch.zeros(1, batch_size, hidden_size, device=device)

def train_compression(data, num_tokens, weights_folder, embedding_dim=64, hidden_size=256, seq_length=64,
                      batch_size=1024, device='cuda', loss_eps=1e-4, lr_patience=10, phase_patience=100, target_ce=1.25,
                      verbose=True, max_boosters=10, activation_type=ActivationType.LEAKY_RELU):
    # Prepare dataset and dataloader
    dataset = SequenceDataset(data, seq_length, batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ensemble = []
    criterion = nn.CrossEntropyLoss()

    current_phase, best_loss = 0, float('inf')
    while current_phase < max_boosters - 1 and best_loss > target_ce:
        if verbose:
            print(f"Starting training phase {current_phase} (adding booster).")

        # Initialize the model for the current phase
        if current_phase == 0:
            model = FirstModel(num_tokens, embedding_dim, hidden_size, activation_type).to(device)
        else:
            model = BoosterModel(num_tokens, embedding_dim, hidden_size, activation_type).to(device)
        model.train()

        # Phase-specific variables
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience, factor=0.5, verbose=verbose)
        hidden_states = [initialize_hidden(batch_size, hidden_size, device) for _ in range(len(ensemble) + 1)]
        phase_best_loss, stagnant_steps = float('inf'), 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.squeeze(0).to(device), y.squeeze(0).to(device)

            # Compute ensemble logits
            ensemble_logits = 0
            with torch.no_grad():
                prev_feature, embed = None, None
                for j in range(current_phase):
                    m = ensemble[j]
                    if j == 0:
                        logits, gru_out, h_new, embed = m(x, hidden_states[j])
                    else:
                        logits, gru_out, h_new = m(embed, prev_feature, hidden_states[j])
                    hidden_states[j] = h_new.detach()
                    ensemble_logits += logits
                    prev_feature = gru_out

            # Forward pass for the current model
            if current_phase == 0:
                logits, gru_out, h_new, embed = model(x, hidden_states[current_phase])
            else:
                logits, gru_out, h_new = model(embed, prev_feature, hidden_states[current_phase])
            hidden_states[current_phase] = h_new.detach()

            total_logits = ensemble_logits + logits
            loss = criterion(total_logits.view(-1, num_tokens), y.view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Logging
            if verbose and batch_idx % 10 == 0:
                print(f"[Phase {current_phase}] Batch {batch_idx} | Lr: {get_current_lr(optimizer)} | Loss: {loss.item():.4f}")

            # Check for improvement or stagnation
            if loss.item() < phase_best_loss - loss_eps:
                phase_best_loss = loss.item()
                stagnant_steps = 0
            else:
                stagnant_steps += 1
                if stagnant_steps == 1:
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                elif stagnant_steps > phase_patience:
                    if verbose:
                        print(f"[Phase {current_phase}] Stagnation reached after {batch_idx} steps. Best loss: {phase_best_loss:.4f}")
                    model.load_state_dict(best_state)
                    break

            # Early stopping if target loss is achieved
            if phase_best_loss <= target_ce:
                if verbose:
                    print(f"[Phase {current_phase}] Target loss achieved: {phase_best_loss:.4f}")
                break

        # Save the model if it improves the overall best loss
        if phase_best_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{current_phase}_weights.pth"))
            best_loss = phase_best_loss
            model.eval()
            ensemble.append(model)
        else:
            break

        current_phase += 1

    print("Training complete.")
    return ensemble

def train(data_path: str, path_to_save_weights: str):
    if torch.cuda.is_available():
        print("CUDA is available! PyTorch can use GPU.")
    else:
        print("CUDA is not available. PyTorch will use CPU.")
    data = np.fromfile(data_path, dtype=np.uint8)
    num_tokens = 256
    if os.path.exists(path_to_save_weights):
        raise FileExistsError(f"The folder '{path_to_save_weights}' already exists.")
    os.makedirs(path_to_save_weights)
    print(f"Folder '{path_to_save_weights}' created successfully.")

    train_compression(data=data, num_tokens=num_tokens, weights_folder=path_to_save_weights)


if __name__ == "__main__":
    train(data_path="enwik8", path_to_save_weights="enwik8_rerun")

