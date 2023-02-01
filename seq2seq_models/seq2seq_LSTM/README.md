# Sequence to Sequence Learning with Neural Networks

Machine translation model architecture:

![](https://github.com/bentrevett/pytorch-seq2seq/blob/master/assets/seq2seq1.png?raw=1)

# Files 

- **model.py**: Definition of the main classes of the model (Encoder, Decoder and final model)
- **data.py**: Datasets classes implementation (only Multi30k for now). Each dataset class builds a iterator for the train, validation and test data, after some pre-processing using `torchtext`.
- **train.py**: Script to train the model on the Multi30k dataset.
- **predict.py**: Loads a pre-trained model and translates the sentences presents in a text file. See [examples](/examples/).
- **analysis.py**: Analysis of the embeddings (sentences and words) quality.
- **seq2seq_training_analysis.py**: Notebook containing the model training and embeddings analysis examples with iterative plots. 

