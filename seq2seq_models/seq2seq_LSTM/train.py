import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import time 
import math
import argparse
from utils.data import Multi30kDataset
from .model import *
from utils.utils import *

def train_model(*args):
    args = parse_args(*args)
    set_seeds(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Chosen device {device}')
    dataset = Multi30kDataset(args.src_lang, args.trg_lang)
    dataset.build(device, batch_size=args.batch_size)
    train_iter = dataset.get_DataLoader('train')
    valid_iter = dataset.get_DataLoader('valid')
    input_dim = dataset.get_vocabsize('src')
    output_dim = dataset.get_vocabsize('trg')
    print(f'Source vocab size: {input_dim}')
    print(f'target vocab size: {output_dim}')

    enc = Encoder(input_dim, args.emb_dim_encoder, args.hid_dim, args.nlayers, args.dropout_enc)
    dec = Decoder(output_dim, args.emb_dim_decoder, args.hid_dim, args.nlayers, args.dropout_dec)
    model = Seq2Seq(enc, dec, device, dataset.get_bos_tensor(), dataset.get_eos_tensor()).to(device)
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
            torch.save(model, args.file)
        
        print(f'\tTime: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    dataset.save_vocabs()

def parse_args(*args):
    parser = argparse.ArgumentParser(description='Seq2Seq model training script')
    parser.add_argument('-src', '--src_lang', type=str, help='source language', required=True)
    parser.add_argument('-trg', '--trg_lang', type=str, help='target language', required=True)
    parser.add_argument('-f', '--file', type=str, help='file path to save the model', required=True)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate for the optimizer')
    parser.add_argument('-ep', '--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size for the dataloader')
    parser.add_argument('--emb_dim_encoder', default=256, type=int, help='encoder embedding dimension')
    parser.add_argument('--emb_dim_decoder', default=256, type=int, help='decoder embedding dimension')
    parser.add_argument('--hid_dim', default=512, type=int, help='hidden state dimension')
    parser.add_argument('-l', '--nlayers', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--dropout_enc', default=0.5, type=float, help='dropout for encoder')
    parser.add_argument('--dropout_dec', default=0.5, type=float, help='dropout for decoder')
    parser.add_argument('-s', '--seed', default=42, type=int, help='seed number for reproducibility')
    return parser.parse_args(*args)

if __name__ == '__main__':
    train_model()
    