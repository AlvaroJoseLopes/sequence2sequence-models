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

def get_word_embeddings(model, module, lang):
    # module = 'encoder' or 'decoder'
    lang_vocab = torch.load(f'vocabs/vocab_{lang}.pth').to(device)
    matrix = model.get_embeddings_matrix(module).cpu().detach().numpy()
    word_embeddings = {}
    for token, embedding in enumerate(list(matrix)):
        word_embeddings[lang_vocab.lookup_token(token)] = embedding

    return word_embeddings 


def parse_args(*args):
    parser = argparse.ArgumentParser(description='Seq2Seq model predict sample script')
    parser.add_argument('-src', '--src_lang', type=str, help='source language', required=True)
    parser.add_argument('-fm', '--file_model', type=str, help='file of the model (.pt)')
    parser.add_argument('-fe', '--file_examples', type=str, help='file of the examples (.txt)')
    return parser.parse_args(*args)

if __name__ == '__main__':
    from data import *
    from model import Seq2Seq
    import pandas as pd

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.file_model).to(device)
    f = open(args.file_examples)
    examples = f.read().splitlines()
    print(examples)
    hd, cell = get_sentences_embeddings(model, examples, args.src_lang, device)
    print(hd)
    print(cell)

    folder = os.path.join('embeddings', '')
    if not os.path.exists(folder):
        os.mkdir(folder)
    word_emb_en = get_word_embeddings(model, 'encoder', 'en')
    df_emb_en = pd.DataFrame.from_dict(word_emb_en, orient='index')
    df_emb_en.to_csv(f'{folder}emb_en.csv')
    word_emb_de = get_word_embeddings(model, 'decoder', 'de')
    df_emb_de = pd.DataFrame.from_dict(word_emb_de, orient='index')
    df_emb_de.to_csv(f'{folder}emb_de.csv')
else:
    from .data import *
    from .model import Seq2Seq