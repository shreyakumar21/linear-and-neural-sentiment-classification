# models.py

import torch
import torch.nn as nn
import nltk 
from nltk.corpus import stopwords
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter
import matplotlib.pyplot as plt
import math


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1
#Dont touch this class^

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        # Call UnigramFeatureExtractor 

        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
        # Call pre-processing function to get rid of stop words
        # Split sentence into a list of words
        # Store indices of positive words in a dictionary
        # return feature vector with indices
    def __init__(self, indexer: Indexer):
        self.indexer=indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        for i in range(len(sentence)):
            sentence[i] = sentence[i].lower()
        count = Counter()
        if add_to_indexer == True:
            for s in sentence:
                if self.indexer.contains(s)==False:
                    idx=self.indexer.add_and_get_index(s)
                else: 
                    idx=self.indexer.index_of(s);
                count[idx]=count[idx]+1
        else:
            for s in sentence:
                if self.indexer.contains(s) == True:
                    idx=self.indexer.index_of(s);
                    count[idx]=count[idx]+1
        return count  

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer=indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        for i in range(len(sentence)):
            sentence[i] = sentence[i].lower()
        count = Counter()
        if add_to_indexer == True:
            for s in zip(sentence, sentence[1:]):
                if self.indexer.contains(s)==False:
                    idx=self.indexer.add_and_get_index(s)
                else: 
                    idx=self.indexer.index_of(s);
                count[idx]=count[idx]+1
        else:
            for s in zip(sentence, sentence[1:]): 
                if self.indexer.contains(s) == True:
                    idx=self.indexer.index_of(s);
                    count[idx]=count[idx]+1
        return count  


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    #OPTIONAL- research later

    def __init__(self, indexer: Indexer):
        self.indexer=indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        stop_words = ['a','an','the','that','with','do', 'there', 'about', 'once', 'during', 'out', 'having','they', 'own', 'be', 'some']
        for i in range(len(sentence)):
            sentence[i] = sentence[i].lower()
        count = Counter()
        if add_to_indexer == True:
            for s in sentence:
                if s in stop_words:
                    continue
                if self.indexer.contains(s)==False:
                    idx=self.indexer.add_and_get_index(s)
                else: 
                    idx=self.indexer.index_of(s);
                count[idx]=count[idx]+1
        else:
            for s in sentence: 
                if s in stop_words:
                    continue
                if self.indexer.contains(s) == True:
                    idx=self.indexer.index_of(s);
                    count[idx]=count[idx]+1
        return count  

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weight_size:int, feat_extractor:FeatureExtractor):
        self.w = np.zeros(weight_size)
        self.fe = feat_extractor

    def probability_calc(self, counter_list:Counter) -> float:
        prod=0
        for idx in counter_list.keys():
            prod+=self.w[idx]*counter_list[idx]
        return math.exp(prod)/(1+math.exp(prod))

    def predict(self, ex_words: List[str]) -> int:
        item = self.fe.extract_features(ex_words, False)
        prob = self.probability_calc(item)
        if prob>0.5:
            return 1
        else:
            return 0

    def train_loss(self, sentences: List[List[str]]) -> float:
        total_loss=0
        n=len(sentences)
        for idx in range(0,n):
            counter_value=self.fe.extract_features(sentences[i].words,False)
            if sentences[idx].label==1:
                if self.probability_calc(counter_value)!=0:
                    val = self.probability_calc(counter_value)
                    total_loss-=math.log(val)
            else:
                if self.probability_calc(counter_value)!=1:
                    val = 1-self.probability_calc(counter_value)
                    total_loss-=math.log(val)
        return total_loss

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return super().predict_all(all_ex_words)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    n = len(train_exs)
    tracker = []
    for idx in range(0,n):
        item = feat_extractor.extract_features(train_exs[idx].words, True)
        tracker.append(item)
    weight_size = len(feat_extractor.indexer)
    lrc_model = LogisticRegressionClassifier(weight_size, feat_extractor)
    training_size = 685000
    for idx in range(training_size):
        i = random.randint(0, n-1)
        c = tracker[i]
        key_update = list(c.keys())
        value_update = list(c.values()) 
        training_label = train_exs[i].label
        probability = training_label - lrc_model.probability_calc(c)
        value_update = [element * probability for element in value_update]
        lrc_model.w[key_update] += value_update
    return lrc_model

def learning_rate_plotting(train_exs: List[SentimentExample], dev_exs: List[SentimentExample], feat_extractor: FeatureExtractor):
    n=len(train_exs)
    tracker= []
    training_loss_values=[]
    dev_accuracy=[]
    feature_extracter= UnigramFeatureExtractor(Indexer())
    for idx in range(0,n):
        counter = feature_extracter.extract_features(train_exs[idx].words, True)
        tracker.append(counter)
    weight_size = len(feat_extractor.indexer)
    lrc_model = LogisticRegressionClassifier(weight_size,feature_extracter)
    training_size = 100
    labels=[s.label for s in dev_exs]
    for i in range(training_size):
        ex_indices=[idx for idx in range(0,n)]
        random.shuffle(ex_indices)

        for idx in ex_indices:
            c = tracker[idx]
            training_label = train_exs[i].label
            probability = training_label - lrc_model.probability_calc(c)
            key_update = list(counter.keys())
            value_update = list(counter.values()) 
            lrc_model.w[key_update] += value_update
            value_update = [element * probability for element in value_update]
            lrc_model.w[key_update] += value_update

        training_loss_values.append(lrc_model.train_loss(train_exs))
        pred_labels= lrc_model.predict_all([s.words for s in dev_exs])
        correct_values=0
        for i in range(len(pred_labels)):
            if pred_labels[i]==labels[i]:
                correct_values+=1
        dev_accuracy.append(correct_values/len(pred_labels))

        plt.plot(np.arange(training_size), dev_accuracy)
        plt.xlabel('Training Size (Epochs)')
        plt.ylabel('Dev accuracy')
        plt.show()

        return

def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network=network
        self.word_embeddings=word_embeddings

    def predict(self, ex_words: List[str]) -> int:
        output = self.network.forward(ex_words)
        if output[0] >= output[1]:
            return 0
        else:
            return 1

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return super().predict_all(all_ex_words)

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    num_classes = 2
    network = DAN(word_embeddings, num_classes)
    network = network.double()
    nsc = NeuralSentimentClassifier(network, word_embeddings)
    num_epochs = 10
    initial_learning_rate = 0.001
    optimizer = optim.Adam(nsc.network.parameters(), lr=initial_learning_rate)
    for epoch in range(num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = train_exs[idx].words
            y = train_exs[idx].label 
            y_onehot = torch.zeros(num_classes)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            y_onehot = y_onehot.double()
            nsc.network.zero_grad()
            log_probs = nsc.network.forward(x)
            loss = torch.neg(log_probs).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("The total loss on Epochs %i: %f" % (epoch, total_loss))
    return nsc


class DAN(nn.Module):
    def __init__(self, word_embeddings, out):
        super(DAN, self).__init__()
        self.word_embeddings = word_embeddings
        self.layer=self.word_embeddings.get_initialized_embedding_layer()
        self.hidden = self.word_embeddings.get_embedding_length() 
        self.W = nn.Linear(self.hidden, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, x):
        n=len(x)
        sum = torch.zeros(self.hidden) 
        for idx in range(0,n):
            sum += self.word_embeddings.get_embedding(x[idx])
        mean = sum/n
        return self.log_softmax(self.W(mean))