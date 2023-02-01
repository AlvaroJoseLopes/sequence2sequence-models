import argparse
import torch
from torchtext.data.utils import get_tokenizer
from utils.data import *
from .model import Seq2Seq

def predict(model, examples, src_lang, trg_lang, device):
    src_tokenizer = get_tokenizer('spacy', language=args.src_lang)
    src_vocab = torch.load(f'vocabs/vocab_{args.src_lang}.pth').to(device)
    trg_vocab = torch.load(f'vocabs/vocab_{args.trg_lang}.pth').to(device)

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

    preds = model.predict(examples)
    preds = torch.transpose(preds, 0, 1)
    preds = [trg_vocab.lookup_tokens([int(token) for token in pred]) for pred in preds.tolist()]
    return [' '.join(pred[1:pred.index('<eos>')]) for pred in preds]

def parse_args(*args):
    parser = argparse.ArgumentParser(description='Seq2Seq model predict sample script')
    parser.add_argument('-src', '--src_lang', type=str, help='source language', required=True)
    parser.add_argument('-trg', '--trg_lang', type=str, help='target language', required=True)
    parser.add_argument('-fm', '--file_model', type=str, help='file of the model (.pt)')
    parser.add_argument('-fe', '--file_examples', type=str, help='file of the examples (.txt)')    
    return parser.parse_args(*args)

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.file_model).to(device)
    f = open(args.file_examples)
    examples = f.read().splitlines()
    print(examples)
    print(predict(model, examples, args.src_lang, args.trg_lang, device))
else:
    from utils.data import *
    from .model import Seq2Seq