import re
from typing import List

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.text import TextCollection

from nltk import *
from nltk.corpus.reader.util import *
import nltk.collocations
from lab_utils import LabPredictor

# pylint: disable=pointless-string-statement
""" Welcome to the first lab!

The comments and TODOs should guide you through the implementation,
but feel free to modify the variables and the overall structure as you see fit.

It is important to ekep the name of the main class: Lab1, as this 
is imported by the `lab_runner.py` file.

You should complete the code for the classes:
- NgramModel (superclass of BigramModel and TrigramModel)
- BigramModel (should be a simple implementation with a few parameters)
- TrigramModel (should be a simple implementation with a few parameters)
- Lab1 (the main logic for parsing input and handling models)
"""

class NgramModel():
    """ The main class for all n-gram models

    Here you will create your model (based on N)
    and complete the predict method to return the most likely words.
    
    """
    def __init__(self, n_gram=2) -> None:
        """ the init method should load/train your model
        Args:
            n_gram (int, optional): 2=bigram, 2=trigram, ... Defaults to 1.
        """
        print(f"Loading {n_gram}-gram model...")
        self.n_gram = n_gram
        self.words_to_return = 7  # how many words to show in the UI
        self.tokens = None

        self.model = BigramModel  # TODO: implement the model using built-in NLTK methods
        # take a look at the nltk.collocations module
        # https://www.nltk.org/howto/collocations.html

    def predict(self, tokens: List[str]) -> List[str]:
        """ given a list of tokens, return the most likely next words

        Args:
            tokens (List[str]): preprocessed tokens from the LabPredictor

        Returns:
            List[str]: selected candidates for next-word prediction
        """
        # we're only interested in the last n-1 words.
        # e.g. for a bigram model,
        # we're only interested in the last word to predict the next
        self.tokens = tokens
        n_tokens = tokens[-(self.n_gram - 1):]


        probabilities = [] # TODO: find the probabilities for the next word(s)


        # TODO: apply some filtering to only select the words
        # here you're free to select your filtering methods
        # a simple approach is to simply sort them by probability

        best_matches = self.model.suggest_next_word(self) # TODO: sort/filter to your liking

        # then return as many words as you've defined above
        return best_matches[:self.words_to_return]


class BigramModel(NgramModel):
    def __init__(self, corpus) -> None:
        super().__init__(n_gram=2)
        self.corpus = corpus
        self.bigrams = self.create_bigrams(corpus)
        self.bigram_counts = nltk.FreqDist(self.bigrams) # TODO: create a bigram count dictionary from the corpus
        self.bigramcount = self.get_bigram_count(self.bigrams)
        self.vocab = set(corpus) # TODO: create a vocabulary from the corpus


    def create_bigrams(self, tokens):
        bigrams = nltk.bigrams(tokens)
        return bigrams
    
    def get_bigram_count(self, bigram):
        freq = nltk.FreqDist(bigram)
        for k, v in freq.items():
            if k == bigram:
                return int(v)
        else:
            return 1
    
    
    # Function takes sentence as input and suggests possible words that comes after the sentence  
    def suggest_next_word(self):
        # Consider the last bigram of sentence
        input_ = self.tokens
        print(input_)
        last_bigram = input_[-1:]
        # Calculating probability for each word in vocab
        vocab_probabilities = {}
        for vocab_word in self.vocab:
            test_bigram = (last_bigram[0], vocab_word)
            
            test_bigram_count = self.bigram_counts.get(test_bigram, 0)

            probability = test_bigram_count / len(self.vocab)
            vocab_probabilities[vocab_word] = probability

        # Sorting the vocab probability in descending order to get top probable words
        top_suggestions = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        get_words = [word for word, prob in top_suggestions]
        return get_words
        
        
    

class TrigramModel(NgramModel):
    corpustext = None

    def __init__(self, corpus) -> None:
        super().__init__(n_gram=3)

"""     # Function takes sentence as input and suggests possible words that comes after the sentence  
    def suggest_next_word(input_, trigram_counts, vocab):
        # Consider the last bigram of sentence
        tokenized_input = nltk.word_tokenize(input_.lower())
        last_bigram = tokenized_input[-2:]
        
        # Calculating probability for each word in vocab
        vocab_probabilities = {}
        for vocab_word in vocab:
            test_trigram = (last_bigram[0], last_bigram[1], vocab_word)
            test_bigram = (last_bigram[0], last_bigram[1])

            test_trigram_count = trigramfreq.get(test_trigram, 0)
            

            probability = test_trigram_count
            vocab_probabilities[vocab_word] = probability
        
        # Sorting the vocab probability in descending order to get top probable words
        top_suggestions = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        return top_suggestions
 """

class Lab1(LabPredictor):
    def __init__(self):
        super().__init__()
        self.unprocessed_corpus = nltk.corpus.gutenberg.words()
        self.processed = self.preprocess(self.unprocessed_corpus) # TODO: load a corpus from NLTK
        self.vocab = set(self.unprocessed_corpus)  # TODO: create a vocabulary from the corpus


        # Define a strategy to select the first words (when there's no input)
        # TODO: this should not be a predefined list
        self.start_words = ["the", "a", "an", "this", "that", "these", "those"]

    @staticmethod
    def preprocess(text) -> List[str]:
        """
        Preprocess the input text as you see fit, return a list of tokens.

        - should you consider parentheses, punctuation?
        - lowercase?
        - find inspiration from the course literature :-)
        """

        # TODO: filters here
        def filter_corpus(text):
            filteredtext = []
            stopwords = nltk.corpus.stopwords.words('english')
            for word in text:
                if word.isalpha() and word and word not in stopwords and len(word) > 1:
                    filteredtext.append(word)
            print(filteredtext[0:100])
            return filteredtext

        return filter_corpus(text)



    def predict(self, input_text):
        if not bool(input_text):  # if there's no input...
            print("No input, using start words")
            return self.start_words

        # make use of the backoff model (e.g. bigram)
        too_few = len(input_text) < 3  # TODO: check if the input is too short for trigrams
        
        tokens = self.preprocess(input_text)

        # select the correct model based on the condition
        model = self.backoff_model 
        # if too_few else self.model
        # alternatively, you can switch between the tri- and bigram models
        # based on the output probabilities. This is 100% optional!

        return model.predict(tokens)

    def train(self) -> None:
        """ train or load the models
        add parameters as you like, such as the corpora you selected.
        """
        print("Training models...")

        self.backoff_model = BigramModel(self.processed)  # TODO: add needed parameters
