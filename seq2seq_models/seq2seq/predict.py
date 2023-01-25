import argparse
from .data import *
from .model import Seq2Seq
import torch
from torchtext.data.utils import get_tokenizer

def predict(*args):
    args = parse_args(*args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(f'models/{args.src_lang}_{args.trg_lang}_model.pt').to(device)
    src_tokenizer = get_tokenizer('spacy', language=args.src_lang)
    src_vocab = torch.load(f'vocabs/vocab_{args.src_lang}.pth').to(device)
    trg_vocab = torch.load(f'vocabs/vocab_{args.trg_lang}.pth').to(device)

    _tokenize = lambda sentence, tokenizer, vocab: \
            [vocab[token] for token in tokenizer(sentence)]
    example = _tokenize(args.example, src_tokenizer, src_vocab)
    example = torch.tensor([[src_vocab['<bos>']] + example + [src_vocab['<eos>']]], dtype=torch.int64, device=device)
    example = example.reshape(example.shape[-1], 1)

    preds = model.predict(example)
    preds = [int(token) for token in list(preds)][1:]
    preds = trg_vocab.lookup_tokens(preds)
    preds = preds[:preds.index('<eos>')]
    print(' '.join(preds))

def parse_args(*args):
    parser = argparse.ArgumentParser(description='Seq2Seq model predict sample script')
    parser.add_argument('-src', '--src_lang', type=str, help='source language', required=True)
    parser.add_argument('-trg', '--trg_lang', type=str, help='target language', required=True)
    parser.add_argument('-ex', '--example', type=str, help="sentence example (len < 30) to predict")    
    return parser.parse_args(*args)

if __name__ == "__main__":
    predict()