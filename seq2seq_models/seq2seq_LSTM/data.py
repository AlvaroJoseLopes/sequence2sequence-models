from collections import Counter
import os
import torch
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence

class Multi30kDataset():
    def __init__(self, src_lang, trg_lang):
        super().__init__()

        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.train_data, self.valid_data, self.test_data = \
            Multi30k(split=('train', 'valid', 'test'),
                    language_pair=(self.src_lang, self.trg_lang))
        self.src_tokenizer = get_tokenizer('spacy', language=self.src_lang)
        self.trg_tokenizer = get_tokenizer('spacy', language=self.trg_lang)
        self.bos_tensor = None
        self.eos_tensor = None
    
    def build(self, device, batch_size=32):
        self.device = device
        self.src_vocab, self.trg_vocab = self._build_vocabs(self.train_data)
        self.bos_tensor = torch.tensor([self.src_vocab['<bos>']], dtype=torch.int64, device=self.device)
        self.eos_tensor = torch.tensor([self.src_vocab['<eos>']], dtype=torch.int64, device=self.device)

        self.train_data = self._tokenize_data(self.train_data, device)
        self.valid_data = self._tokenize_data(self.valid_data, device)
        # self.test_data = self._tokenize_data(self.test_data, device)

        self.pad_idx = self.src_vocab['<pad>']
        self.train_iter = DataLoader(
            self.train_data, batch_size=batch_size, collate_fn=self._generate_batch
        )
        self.valid_iter = DataLoader(
            self.valid_data, batch_size=batch_size, collate_fn=self._generate_batch
        )
        # self.test_iter = DataLoader(
        #     self.test_data, batch_size=batch_size, collate_fn=self._generate_batch
        # )
    
    def get_DataLoader(self, dataloader_type):
        try:
            dataloader_map = {
                'train': self.train_iter,
                'valid': self.valid_iter,
                # 'test': self.test_iter
            }
            return dataloader_map[dataloader_type]
        except:
            raise Exception(f"Could not access the dataloader variables\n\
                            Make sure to run the \'build\' method before")
    
    def get_vocabsize(self, vocab_type):
        try:
            vocabsize_map = {
                'src': len(self.src_vocab),
                'trg': len(self.trg_vocab)
            }
            return vocabsize_map[vocab_type]
        except:
            raise Exception(f"Could not access the vocab variables\n\
                            Make sure to run the \'build\' method before")

    def get_pad_idx(self):
        return self.pad_idx
    
    def get_bos_tensor(self):
        return self.bos_tensor
    
    def get_eos_tensor(self):
        return self.eos_tensor

    def save_vocabs(self):
        folder = os.path.join('vocabs', '')
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save(self.src_vocab, f'{folder}vocab_{self.src_lang}.pth')
        torch.save(self.trg_vocab, f'{folder}vocab_{self.trg_lang}.pth')

    def _build_vocabs(self, data):
        data = iter(data)
        src_counter = Counter()
        trg_counter = Counter()
        for (en_sentence, de_sentence) in data:
            src_counter.update(self.src_tokenizer(en_sentence))
            trg_counter.update(self.trg_tokenizer(de_sentence))
        
        src_vocab = vocab(src_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        trg_vocab = vocab(trg_counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

        src_vocab.set_default_index(src_vocab['<unk>'])
        trg_vocab.set_default_index(trg_vocab['<unk>'])
        
        return src_vocab, trg_vocab 

    def _tokenize_data(self, data, device):
        _tokenize = lambda sentence, tokenizer, vocab: \
            [vocab[token] for token in tokenizer(sentence)]
        
        data = iter(data)
        tokenized_data = []
        for (src_sentence, trg_sentence) in data:
            en_tensor = torch.tensor(
                _tokenize(src_sentence, self.src_tokenizer, self.src_vocab),
                dtype=torch.int64, device=device
            )
            de_tensor = torch.tensor(
                _tokenize(trg_sentence, self.trg_tokenizer, self.trg_vocab),
                dtype=torch.int64, device=device
            )
            tokenized_data.append((en_tensor, de_tensor))

        return tokenized_data

    def _generate_batch(self, data):

        src_batch, trg_batch = [], []
        for (src_sentence, trg_sentence) in data:
            src_batch.append(torch.cat([self.bos_tensor, src_sentence, self.eos_tensor], dim=0))
            trg_batch.append(torch.cat([self.bos_tensor, trg_sentence, self.eos_tensor], dim=0))
        
        src_batch = pad_sequence(src_batch, padding_value=self.pad_idx)
        trg_batch = pad_sequence(trg_batch, padding_value=self.pad_idx)
        
        return src_batch, trg_batch
    