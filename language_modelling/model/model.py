import torch.nn as nn


class RNNLM(nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers, dropout, tie_weights):

        super(RNNLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embed_size, hidden_size, num_layers, dropout=dropout)

        self.decoder = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if hidden_size != embed_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        # ___ house keeping_____ #
        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def reset_hidden_states(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded, hidden
