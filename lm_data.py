""" Module for handling test data without test files """

import torch
import os

#from nltk import sent_tokenize

def isfloat(instr):
    """ Reports whether a string is floatable """
    try:
        _ = float(instr)
        return(True)
    except:
        return(False)

class Dictionary(object):
    """ Maps between observations and indices """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """ Adds a new obs to the dictionary if needed """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

#Wrapper for just handling test data as sents (i.e. no test files needed) 
class TestSent:

    def __init__(self, path, vocab_file, sents, multisentence_test_flag=False):
        self.dictionary = Dictionary()
        self.load_dict(vocab_file)
        #make all things lower 
        self.lower = True

        self.word_ids = []
        if multisentence_test_flag:
            all_words = []
            for sent_index in range(len(sents)):
                sent = sents[sent_index]
                if sent_index == 0:
                    words = ['<eos>'] + sent.split() + ['<eos>']
                else:
                    words = sent.split() + ['<eos>']
                all_words += words
            self.word_ids.append(self.convert_to_ids(all_words))
        else:
            for sent in sents:
                words = ['<eos>'] + sent.split() + ['<eos>']
                ids = self.convert_to_ids(words)
                self.word_ids.append(ids)

    def get_data(self):
        return self.word_ids

    def convert_to_ids(self, words, tokens=None):
        if tokens is None:
            tokens = len(words)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        token = 0
        if self.lower:
            for word in words:
                # Convert OOV to <unk>
                if word.lower() not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                token += 1
        else:
            for word in words:
                # Convert OOV to <unk>
                if word not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word]
                token += 1
        return(ids)

    def tokenize_with_unks(self, sent):
        """ Given a sentence return ids, adding unks if needed. """
        words = ['<eos>'] + sent.split() + ['<eos>']
        return self.convert_to_ids(words)
    
    def continued_tokenize_with_unks(self, sent):
        """ Given a non-initial sentence return ids, adding unks if needed. """
        words = sent.split() + ['<eos>']
        return self.convert_to_ids(words)

    def load_dict(self, path):
        """ Loads dictionary from disk """
        assert os.path.exists(path), "Bad path: %s" % path
        # Assume dict is plaintext
        with open(path, 'r') as file_handle:
            for line in file_handle:
                self.dictionary.add_word(line.strip())
