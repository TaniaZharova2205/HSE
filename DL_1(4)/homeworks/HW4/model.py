import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """

        device = next(self.parameters()).device
        indices = indices.to(device)
        lengths = lengths.to(device)
        embedded = self.embedding(indices)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        # алгоритм взят из https://www.cs.toronto.edu/~lczhang/360/lec/w08/gen.html
        device = next(self.parameters()).device
        
        tokens = [self.dataset.bos_id] + self.dataset.text2ids(prefix)
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        hidden = None
        if isinstance(self.rnn, nn.LSTM):
            # Для LSTM:
            h = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
            c = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
            hidden = (h, c)
        elif isinstance(self.rnn, nn.RNN):
            # Для RNN:
            hidden = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
        
        embedded = self.embedding(input_ids)  
        _, hidden = self.rnn(embedded, hidden)
        
        generated = tokens.copy()
        next_token = generated[-1]
        
        for _ in range(self.max_length - len(generated)):
            inp = torch.tensor([[next_token]], dtype=torch.long, device=device)
            emb = self.embedding(inp)  
            out, hidden = self.rnn(emb, hidden)
            logits = self.linear(out[:, -1, :]) / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if next_token == self.dataset.eos_id:
                break
        generated_text = self.dataset.ids2text(generated[1:])
        return generated_text