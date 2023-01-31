import torch
import torch.nn as nn
import random 

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input.shape = (sequence_len, batch_size)
        
        # embedding.shape = (sequence_len, emb_dim, batch_size)
        embeddings = self.dropout(self.embedding(input))

        # output.shape = (sequence_len, batch_size, n_directions * hid_dim)
        # h_n.shape = (n_directions * n_layers, batch_size, hid_dim) -> hidden states
        # c_n.shape = (n_directions * n_layers, batch_size, hid_dim) -> cell states
        _, (h_n, c_n) = self.rnn(embeddings)
        
        return h_n, c_n

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim 
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        # output_dim will be the size of the vocab
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_dim, output_dim) 
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        # input.shape = (batch_size)
        #   sentence_len = 1 because each token at a time will be decoded
        # hidden.shape = (n_directions * n_layers, batch_size, hid_dim)
        # cell.shape = (n_directions * n_layers, batch_size, hid_dim)

        # input.shape = (1, batch_size)
        input = input.unsqueeze(0)

        # embedding.shape = (1, batch_size, emb_dim)
        embedding = self.dropout(self.embedding(input))

        # output.shape = (sequence_len, batch_size, n_directions * hid_dim)
        # h_n.shape = (n_directions * n_layers, batch_size, hid_dim) -> hidden states
        # c_n.shape = (n_directions * n_layers, batch_size, hid_dim) -> cell states
        # note: sequence_len=1
        output, (h_n, c_n) = self.rnn(embedding, (hidden, cell))

        # prediction.shape = (batch_size, output_dim)
        prediction = self.out(output.squeeze(0))

        return prediction, h_n, c_n

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, bos_tensor, eos_tensor, max_len=30):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.bos_tensor = bos_tensor
        self.eos_tensor = eos_tensor
        self.max_len = max_len

        assert encoder.hid_dim == decoder.hid_dim, "hid_dim must be equal for encoder and decoder"
        assert encoder.n_layers == decoder.n_layers, "n_layers must be equal for encoder and decoder"

    def forward(self, source, target, teacher_forcing_rate=0.5):
        # src.shape = (src_len, batch_size)
        # target.shape = (target_len, batch_size)

        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_dim 

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
        
        # last hidden and cell states will be the initial hidden/cell state of the decoder
        hidden, cell = self.encoder(source)
        # first input to the decoder is the <bos> token
        input = self.bos_tensor.repeat(batch_size)

        for pred_idx in range(1, target_len):
            # receive the prediction (output) and the next hidde/cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # stores the prediction on the final output
            outputs[pred_idx] = output

            # get the next input using teacher forcing
            teacher_force = random.random() < teacher_forcing_rate
            pred = output.argmax(1)
            input = target[pred_idx] if teacher_force else pred
        
        return outputs
    
    def predict(self, example):
        self.eval()
        with torch.no_grad():

            batch_size = example.shape[1]

            # outputs = torch.zeros(self.max_len, batch_size, target_vocab_size).to(self.device)
            preds = torch.zeros(self.max_len, batch_size).to(self.device)
            preds[0] = example[0,:]
        
            # last hidden and cell states will be the initial hidden/cell state of the decoder
            hidden, cell = self.encoder(example)
            # first input to the decoder is the <bos> token
            input = self.bos_tensor.repeat(batch_size)

            for pred_idx in range(1, self.max_len):
                output, hidden, cell = self.decoder(input, hidden, cell)
            
                # outputs[pred_idx] = output
                preds[pred_idx] = output.argmax(1)
                input = output.argmax(1)

                # if input == self.eos_tensor:
                #     break
        
        return preds

    def encode(self, examples):
        self.eval()
        with torch.no_grad():
            return  self.encoder(examples)

    def get_embeddings_matrix(self, lang):
        embeddings_mapper = {
            'encoder': self.encoder.embedding.weight,
            'decoder': self.decoder.embedding.weight
        }
        return embeddings_mapper[lang]