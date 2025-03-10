import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# =============================================================================
# DataSet definition: creates B independent sequences from a long 1D numpy array.
# =============================================================================
class SequenceDataset(Dataset):
    def __init__(self, data, seq_length, batch_size, in_memory=True):
        """
        data: 1D numpy array of integer tokens.
        seq_length: length L of each training sequence.
        batch_size: B independent sequences.
        in_memory: if True, assume data fits in memory.
        """
        self.seq_length = seq_length
        self.batch_size = batch_size
        N = len(data)
        # If possible, we reshape the data into (B, T) where T = floor(N/B)
        T = N // batch_size
        self.data = data[:batch_size * T].reshape(batch_size, T)
        # We need a target sequence which is shifted by one.
        # How many batches can we extract? (we lose one time step for target)
        self.num_batches = (T - 1) // seq_length

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # idx selects a segment along each of the B sequences.
        start = idx * self.seq_length
        end = (idx + 1) * self.seq_length
        # Input: positions [start:end], Target: positions [start+1:end+1]
        x = self.data[:, start:end]
        y = self.data[:, start + 1:end + 1]
        # Convert to torch tensors
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# =============================================================================
# Base Predictor: GRU + skip connection + layer norm + dense.
# Note: the GRU input dimension is set by whatever “input_dim” is provided.
# =============================================================================
class BasePredictor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_tokens):
        """
        input_dim: dimension of the input features.
        hidden_size: GRU hidden size (also the output feature size).
        num_tokens: number of classes (vocabulary size).
        """
        super(BasePredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        # A linear layer to map input (for skip connection) to hidden size.
        self.skip_proj = nn.Linear(input_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_tokens)

    def forward(self, x, hidden=None):
        # x: (B, L, input_dim)
        gru_out, hidden_new = self.gru(x, hidden)  # gru_out: (B, L, hidden_size)
        skip = self.skip_proj(x)
        out = gru_out + skip
        out = self.layer_norm(out)
        logits = self.fc(out)  # (B, L, num_tokens)
        return logits, out, hidden_new


# =============================================================================
# FirstModel: includes an embedding and a BasePredictor.
# For phase 0 training, no booster input is required.
# =============================================================================
class FirstModel(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_size):
        super(FirstModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        # The predictor takes the embedding as input.
        self.predictor = BasePredictor(input_dim=embedding_dim, hidden_size=hidden_size, num_tokens=num_tokens)

    def forward(self, x, hidden=None):
        # x: (B, L) token indices.
        embed = self.embedding(x)  # (B, L, embedding_dim)
        logits, gru_out, hidden_new = self.predictor(embed, hidden)
        return logits, gru_out, hidden_new


# =============================================================================
# BoosterModel: for subsequent phases, the new model gets as input a concatenation
# of the original embedding and the previous model's GRU output.
# =============================================================================
class BoosterModel(nn.Module):
    def __init__(self, num_tokens, embedding_dim, hidden_size):
        """
        In this example we assume that the booster model will receive:
           - the original embedding (dimension embedding_dim)
           - the previous phase's GRU output (dimension hidden_size)
        so the effective input dim is (embedding_dim + hidden_size).
        """
        super(BoosterModel, self).__init__()
        self.predictor = BasePredictor(input_dim=embedding_dim + hidden_size,
                                       hidden_size=hidden_size,
                                       num_tokens=num_tokens)

    def forward(self, embed, prev_feature, hidden=None):
        # embed: (B, L, embedding_dim); prev_feature: (B, L, hidden_size)
        x = torch.cat([embed, prev_feature], dim=-1)
        logits, gru_out, hidden_new = self.predictor(x, hidden)
        return logits, gru_out, hidden_new


# =============================================================================
# CompressionTrainer: organizes the training phases and ensemble boosting.
# =============================================================================
class CompressionTrainer:
    def __init__(self, data, num_tokens, embedding_dim=64, hidden_size=128,
                 seq_length=100, batch_size=32, device='cuda', loss_eps=1e-4,
                 patience=100, target_ce=1.25, verbose=True):
        """
        data: numpy 1D ndarray of integer tokens.
        num_tokens: vocabulary size.
        embedding_dim: dimension of embedding vectors.
        hidden_size: GRU hidden size.
        seq_length: L - length of each sequence chunk.
        batch_size: B - number of independent sequences.
        device: 'cuda' or 'cpu'.
        loss_eps: minimum improvement in loss to be considered progress.
        patience: number of steps to wait on stagnation before stopping current phase.
        target_ce: target cross entropy loss to stop overall training.
        verbose: if True, print progress.
        """
        self.device = device
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.loss_eps = loss_eps
        self.patience = patience
        self.target_ce = target_ce
        self.verbose = verbose

        # Create the dataset and DataLoader.
        self.dataset = SequenceDataset(data, seq_length, batch_size)
        # Each __getitem__ returns a full batch of shape (B, L) so we use batch_size=1 here.
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        # Initialize ensemble with the phase 0 model.
        self.first_model = FirstModel(num_tokens, embedding_dim, hidden_size).to(device)
        # The ensemble is a list of models. At phase 0 the ensemble contains only first_model.
        self.ensemble = [self.first_model]
        # For each model we maintain its hidden state (initialized later as zeros).
        self.hidden_states = [None]
        # Loss criterion: PyTorch’s CrossEntropyLoss expects raw logits and integer targets.
        self.criterion = nn.CrossEntropyLoss()

    def _init_hidden(self, batch_size):
        # Initialize hidden state for GRU: shape (num_layers, B, hidden_size).
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def train_phase(self, model, phase_idx, max_epochs=10):
        """
        Trains the given model (phase) while keeping the ensemble prior to phase_idx frozen.
        Only the current model’s parameters are updated.
        Returns the best achieved loss.
        """
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # A scheduler that reduces lr on plateau.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, self.patience // 10), factor=0.5,
                                                         verbose=self.verbose)
        best_loss = float('inf')
        stagnant_steps = 0
        step_count = 0

        # Initialize hidden state for current model if not yet set.
        if self.hidden_states[phase_idx] is None:
            self.hidden_states[phase_idx] = self._init_hidden(self.batch_size)

        # Loop over epochs and batches.
        for epoch in range(1, max_epochs + 1):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                # x and y are of shape (1, B, L) because DataLoader batches the dataset;
                # squeeze out the extra dimension.
                x = x.squeeze(0).to(self.device)  # (B, L)
                y = y.squeeze(0).to(self.device)  # (B, L)

                # First, run the frozen ensemble (all phases before current phase) in eval mode.
                ensemble_logits = 0
                with torch.no_grad():
                    prev_feature = None
                    for j in range(phase_idx):
                        m = self.ensemble[j]
                        m.eval()
                        # For phase 0 model the input is just the token indices.
                        if j == 0:
                            # If hidden state not initialized, do so.
                            if self.hidden_states[j] is None:
                                self.hidden_states[j] = self._init_hidden(self.batch_size)
                            logits, gru_out, h_new = m(x, self.hidden_states[j])
                            # Update hidden state (detach to prevent backprop).
                            self.hidden_states[j] = h_new.detach()
                            ensemble_logits = ensemble_logits + logits
                            prev_feature = gru_out  # store output for next booster
                        else:
                            # Booster models require the original embedding and previous booster output.
                            m.eval()
                            embed = self.first_model.embedding(x)
                            logits, gru_out, h_new = m(embed, prev_feature, self.hidden_states[j])
                            self.hidden_states[j] = h_new.detach()
                            ensemble_logits = ensemble_logits + logits
                            prev_feature = gru_out

                # Now compute current model output.
                model.train()
                if phase_idx == 0:
                    # For phase 0 model: input is token indices.
                    logits, gru_out, h_new = model(x, self.hidden_states[phase_idx])
                    self.hidden_states[phase_idx] = h_new.detach()
                else:
                    # Booster: input is concatenation of embedding and previous phase’s output.
                    embed = self.first_model.embedding(x)
                    logits, gru_out, h_new = model(embed, prev_feature, self.hidden_states[phase_idx])
                    self.hidden_states[phase_idx] = h_new.detach()

                # The ensemble output is the sum of frozen ensemble logits and current model logits.
                total_logits = ensemble_logits + logits

                # Reshape to (B * L, num_tokens) and target to (B * L)
                loss = self.criterion(total_logits.view(-1, self.num_tokens), y.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step_count += 1
                scheduler.step(loss)

                if self.verbose and step_count % 10 == 0:
                    print(f"[Phase {phase_idx}] Epoch {epoch} Batch {batch_idx} | Loss: {loss.item():.4f}")

                # Check for loss improvement.
                if loss.item() < best_loss - self.loss_eps:
                    best_loss = loss.item()
                    stagnant_steps = 0
                    # Save best state.
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    stagnant_steps += 1

                # If the loss stagnates for too many steps, stop current phase.
                if stagnant_steps > self.patience:
                    if self.verbose:
                        print(
                            f"[Phase {phase_idx}] Stagnation reached after {step_count} steps. Best loss: {best_loss:.4f}")
                    # Restore the best state.
                    model.load_state_dict(best_state)
                    return best_loss

                # Optionally, if loss is low enough, we can break early.
                if best_loss <= self.target_ce:
                    if self.verbose:
                        print(f"[Phase {phase_idx}] Target loss achieved: {best_loss:.4f}")
                    return best_loss

        return best_loss

    def train(self, max_boosters=5, max_epochs_per_phase=10):
        """
        Trains the ensemble phase by phase. In phase 0 the first model is trained.
        Then for each new booster added, the previously trained models are frozen and only
        the new model’s parameters are updated.
        Training stops when either the target cross entropy is reached or the maximum number
        of boosters has been added.
        Returns the ensemble list.
        """
        print("Starting training phase 0 (base model).")
        best_loss = self.train_phase(self.ensemble[0], phase_idx=0, max_epochs=max_epochs_per_phase)
        print(f"Phase 0 complete. Loss: {best_loss:.4f}")

        current_phase = 0
        while current_phase < max_boosters - 1 and best_loss > self.target_ce:
            current_phase += 1
            print(f"Starting training phase {current_phase} (adding booster).")
            # Create new booster model.
            booster = BoosterModel(self.num_tokens, self.embedding_dim, self.hidden_size).to(self.device)
            # Freeze previous models (by not including them in the optimizer) and add the new booster.
            self.ensemble.append(booster)
            self.hidden_states.append(self._init_hidden(self.batch_size))
            phase_loss = self.train_phase(booster, phase_idx=current_phase, max_epochs=max_epochs_per_phase)
            print(f"Phase {current_phase} complete. Loss: {phase_loss:.4f}")
            best_loss = phase_loss

        print("Training complete.")
        return self.ensemble


# =============================================================================
# Example usage
# =============================================================================
if __name__ == '__main__':
    # Create synthetic data for demonstration:
    # Suppose our tokens are in range [0, num_tokens-1].
    num_tokens = 256  # example limited set size
    N = 100000  # length of the 1D numpy array (adjust as needed)
    np.random.seed(42)
    data = np.random.randint(0, num_tokens, size=(N,))

    # Training parameters:
    embedding_dim = 64
    hidden_size = 128
    seq_length = 50
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_eps = 1e-4
    patience = 50
    target_ce = 1.25
    verbose = True

    trainer = CompressionTrainer(data, num_tokens, embedding_dim, hidden_size,
                                 seq_length, batch_size, device, loss_eps,
                                 patience, target_ce, verbose)
    # Train the ensemble (max_boosters sets the maximum number of phases).
    ensemble = trainer.train(max_boosters=5, max_epochs_per_phase=5)
