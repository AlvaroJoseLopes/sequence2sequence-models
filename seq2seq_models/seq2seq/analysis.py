import argparse
import torch
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

def get_sentences_embeddings(model, examples, src_lang, device):
    src_tokenizer = get_tokenizer('spacy', language=src_lang)
    src_vocab = torch.load(f'vocabs/vocab_{src_lang}.pth').to(device)

    _tokenize = lambda sentence, tokenizer, vocab: \
            [vocab[token] for token in tokenizer(sentence)]

    examples = [
            torch.tensor([src_vocab['<bos>']] 
            + _tokenize(example, src_tokenizer, src_vocab) 
            + [src_vocab['<eos>']], dtype=torch.int64, device=device)
        for example in examples
    ]
    examples = pad_sequence(examples, padding_value=src_vocab['<pad>'])
    examples = examples.clone().detach()

    return model.encode(examples)

def parse_args(*args):
    parser = argparse.ArgumentParser(description='Seq2Seq model predict sample script')
    parser.add_argument('-src', '--src_lang', type=str, help='source language', required=True)
    parser.add_argument('-fm', '--file_model', type=str, help='file of the model (.pt)')
    parser.add_argument('-fe', '--file_examples', type=str, help='file of the examples (.txt)')
    return parser.parse_args(*args)

if __name__ == '__main__':
    from data import *
    from model import Seq2Seq

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.file_model).to(device)
    f = open(args.file_examples)
    examples = f.read().splitlines()
    print(examples)
    hd, cell = get_sentences_embeddings(model, examples, args.src_lang, device)
    print(hd)
    print(cell)

else:
    from .data import *
    from .model import Seq2Seq