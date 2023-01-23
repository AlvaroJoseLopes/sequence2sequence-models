import torch
import torch.nn as nn
import random 
import numpy as np
import argparse

# training related functions 

def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, data, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, (source, target) in enumerate(data):
        optimizer.zero_grad()
        output = model(source, target)
        output_dim = output.shape[-1]

        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        output = output[1:].view(-1, output_dim)
        target = target[1:].view(-1)

        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss/len(data)

def evaluate(model, data, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (source, target) in enumerate(data):


            output = model(source, target, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = target[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(data)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Parser related functions

def parse_args():
    parser = argparse.ArgumentParser(description='Seq2Seq model training script')
    parser.add_argument('-src', '--src_lang', type=str, help='source language', required=True)
    parser.add_argument('-trg', '--trg_lang', type=str, help='target language', required=True)
    parser.add_argument('-f', '--file', type=str, help='file path to save the model', required=True)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate for the optimizer')
    parser.add_argument('-ep', '--epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--emb_dim_encoder', default=256, type=int, help='encoder embedding dimension')
    parser.add_argument('--emb_dim_decoder', default=256, type=int, help='decoder embedding dimension')
    parser.add_argument('--hid_dim', default=512, type=int, help='hidden state dimension')
    parser.add_argument('-l', '--nlayers', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--dropout_enc', default=0.5, type=float, help='dropout for encoder')
    parser.add_argument('--dropout_dec', default=0.5, type=float, help='dropout for decoder')
    parser.add_argument('-s', '--seed', default=42, type=int, help='seed number for reproducibility')
    return parser.parse_args()

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True