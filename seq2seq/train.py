import torch
from data import Multi30kDataset
from model import *
from utils import *
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import time 
import math

def main():
    args = parse_args()
    set_seeds(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Chosen device {device}')
    dataset = Multi30kDataset(args.src_lang, args.trg_lang)
    dataset.build(device)
    train_iter = dataset.get_DataLoader('train')
    valid_iter = dataset.get_DataLoader('valid')
    input_dim = dataset.get_vocabsize('src')
    output_dim = dataset.get_vocabsize('trg')
    print(f'Source vocab size: {input_dim}')
    print(f'target vocab size: {output_dim}')

    enc = Encoder(input_dim, args.emb_dim_encoder, args.hid_dim, args.nlayers, args.dropout_enc)
    dec = Decoder(output_dim, args.emb_dim_decoder, args.hid_dim, args.nlayers, args.dropout_dec)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = Adam(model.parameters())
    criterion = CrossEntropyLoss(ignore_index=dataset.get_pad_idx())

    N_EPOCHS = args.epochs
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        print(f'Epoch: {epoch+1:02} ')
        start_time = time.time()
        
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')
        
        print(f'\tTime: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    torch.save(model, args.file)
    dataset.save_vocabs()


if __name__ == "__main__":
    main()


