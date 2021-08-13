from torchtext import data
import spacy
from torchtext.datasets import LanguageModelingDataset
from spacy.symbols import ORTH
import torch
import logging

SOS_WORD = '<sos>'
EOS_WORD = '<eos>'
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'


class DataPreprocessor(object):
    def __init__(self, device, logger):
        self.text_field = self.generate_fields()
        self.device = device
        self.logger = logger

    @ staticmethod
    def generate_fields():
        en_tokenizer = spacy.load('en')
        en_tokenizer.tokenizer.add_special_case(SOS_WORD, [{ORTH: SOS_WORD}])
        en_tokenizer.tokenizer.add_special_case(EOS_WORD, [{ORTH: EOS_WORD}])
        en_tokenizer.tokenizer.add_special_case(PAD_WORD, [{ORTH: PAD_WORD}])
        en_tokenizer.tokenizer.add_special_case(UNK_WORD, [{ORTH: UNK_WORD}])

        def text_tok(x):
            return [tok.text for tok in en_tokenizer.tokenizer(x)]

        text_field = data.Field(tokenize=text_tok, lower=True, include_lengths=False, batch_first=False,
                                pad_token=PAD_WORD)

        return text_field

    def preprocess(self, data_path, train_file, val_file, test_file, train_file_save, val_file_save, test_file_save,
                   save_path, min_freq=5):

        # Generating torchtext dataset class
        self.logger.info("Preprocessing train, validation and test dataset...")
        train_dataset, val_dataset, test_dataset = LanguageModelingDataset.splits(
            path=data_path,
            text_field=self.text_field,
            train=train_file,
            validation=val_file,
            test=test_file)

        self.logger.info("Saving train dataset...")
        self.save_data(save_path + train_file_save, train_dataset, 'text')
        self.logger.info("Saving validation dataset...")
        self.save_data(save_path + val_file_save, val_dataset, 'text')
        self.logger.info("Saving test dataset...")
        self.save_data(save_path + test_file_save, test_dataset, 'text')
        self.logger.info("train,dev and test files have been created")

        # Building field vocabulary
        self.text_field.build_vocab(train_dataset, min_freq=min_freq)
        vocab, inv_vocab = self.generate_vocabs()
        vocabs = {'vocab': vocab, 'inv_vocab': inv_vocab}
        self.save_data(data_file=save_path + 'dictionary', dataset=vocabs)
        self.logger.info("dictionary have been created")

    @ staticmethod
    def save_data( data_file, dataset, type=None):
        if type == 'text':
            examples = vars(dataset)['examples']
            dataset = {'examples': examples}
            torch.save(obj=dataset, f=data_file)
        else:
            torch.save(obj=dataset, f=data_file)

    def generate_vocabs(self):
        # Define string to index vocabs
        vocab = self.text_field.vocab.stoi
        # Define index to string vocabs
        inv_vocab = self.text_field.vocab.itos
        return vocab, inv_vocab

    def load_data(self, train_file, val_file, test_file, path):
        # Loading saved data
        self.logger.info("loading train dataset")
        train_dataset = torch.load(path + train_file)

        self.logger.info("loading dev dataset")
        val_dataset = torch.load(path + val_file)

        self.logger.info("loading test dataset")
        test_dataset = torch.load(path + test_file)

        self.logger.info("loading dictionary")
        vocabs = torch.load(path + "dictionary")

        return train_dataset['examples'], val_dataset['examples'], test_dataset['examples'], vocabs

    def return_iterators(self, train_dataset, val_dataset, test_dataset, batch_size, seq_length, min_freq=5):
        fields = [('text', self.text_field)]
        train_dataset = data.Dataset(fields=fields, examples=train_dataset)
        val_dataset = data.Dataset(fields=fields, examples=val_dataset)
        test_dataset = data.Dataset(fields=fields, examples=test_dataset)
        self.text_field.build_vocab(train_dataset, min_freq=min_freq)

        train_iter, valid_iter, test_iter = data.BPTTIterator.splits(
            (train_dataset, val_dataset, test_dataset), batch_size=batch_size, bptt_len=seq_length, repeat=False,
            device=self.device)

        return train_iter, valid_iter, test_iter


# -------------TESTING------------- #
def main(create_files):

    experiment_name = "Data_loader_testing"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='log/%s' % experiment_name, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    obj = DataPreprocessor('cpu', logging)
    DATA_PATH = '/Users/najamzaidi/projects/PhD/diff_inference_scheme/implementation/data/wikitext-103/raw_data/'
    SAVE_PATH = '/Users/najamzaidi/projects/PhD/diff_inference_scheme/implementation/data/wikitext-103/model_data/'

    # DO IT ONCE TO GENERATE THE FILES
    if create_files:
        obj.preprocess(data_path=DATA_PATH,
                       train_file='train',
                       val_file='dev',
                       test_file='test',
                       train_file_save='train', val_file_save='dev', test_file_save='test',
                       save_path=SAVE_PATH,
                       min_freq=5)

    train_dataset, val_dataset, test_dataset, vocabs = obj.load_data('train', 'dev', 'test', SAVE_PATH)
    train_iter, val_iter, test_iter = obj.return_iterators(train_dataset, val_dataset, test_dataset,
                                                           batch_size=1, seq_length=50, min_freq=5)
    vocab_size = len(vocabs['vocab'])

    print("[vocab size]:%d " % vocab_size)
    print(len(test_iter))
    for i, batch in enumerate(test_iter):
        print('batch number is: ' + str(i))
        s = []
        for sentence in batch.text:
            s += [vocabs['inv_vocab'][word] for word in sentence]
        print(" ".join(s))


if __name__ == '__main__':
    main(False)




