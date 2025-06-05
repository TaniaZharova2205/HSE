import torch
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm.notebook import tqdm
from model import LanguageModel


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses: List[float], val_losses: List[float]):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_ylabel('loss')

    """
    YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
    Calculate train and validation perplexities given lists of losses
    """
    train_perplexities = [torch.exp(torch.tensor(l)).item() for l in train_losses]
    val_perplexities = [torch.exp(torch.tensor(l)).item() for l in val_losses]

    axs[1].plot(range(1, len(train_perplexities) + 1), train_perplexities, label='train')
    axs[1].plot(range(1, len(val_perplexities) + 1), val_perplexities, label='val')
    axs[1].set_ylabel('perplexity')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0
    total_tokens = 0
    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Process one training step: calculate loss,
        call backward and make one optimizer step.
        Accumulate sum of losses for different batches in train_loss
        """
        indices = indices.to(device)
        lengths = lengths.to(device)
        logits = model(indices[:, :-1], lengths - 1)
        targets = indices[:, 1:] 
        loss = 0
        batch_size = indices.size(0)
        for i in range(batch_size):
            # Get the valid portion of the sequence (excluding padding)
            valid_length = (lengths - 1)[i]
            if valid_length <= 0:
                continue
            seq_logits = logits[i, :valid_length]
            seq_targets = targets[i, :valid_length]
            loss += criterion(seq_logits, seq_targets)
            total_tokens += valid_length.item()
        loss = loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size
    train_loss /= total_tokens if total_tokens > 0 else 1
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0
    total_tokens = 0
    
    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Process one validation step: calculate loss.
        Accumulate sum of losses for different batches in val_loss
        """
        indices = indices.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        logits = model(indices[:, :-1], lengths - 1)
        targets = indices[:, 1:]
        
        # Calculate the actual sequence lengths (without BOS/EOS)
        seq_lengths = lengths - 1
        
        # Calculate loss for each sequence up to its actual length
        batch_loss = 0
        batch_size = indices.size(0)
        batch_tokens = 0
        for i in range(batch_size):
            # Get the valid portion of the sequence (excluding padding)
            valid_length = seq_lengths[i]
            if valid_length <= 0:
                continue
                
            # Calculate loss for this sequence
            seq_logits = logits[i, :valid_length]
            seq_targets = targets[i, :valid_length]
            batch_loss += criterion(seq_logits, seq_targets).item()
            batch_tokens += valid_length.item()
        
        if batch_tokens > 0:
            val_loss += batch_loss
            total_tokens += batch_tokens

    val_loss /= total_tokens if total_tokens > 0 else 1
    return val_loss


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        plot_losses(train_losses, val_losses)

        print('Generation examples:')
        for _ in range(num_examples):
            print(model.inference())
