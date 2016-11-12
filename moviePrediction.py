from __future__ import division
import math
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import string

# -------------------------------------------classifier for review------------------------------------------------------
# stopwords and punctuation
stop = set(stopwords.words()).union(set(string.punctuation))
# remove too common words
stop.update(['the', 'and', 'a', 'of', 'to', 'is', 'in', 'i', 'it', 'that'])  # to do

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = r"E:\cs585project\NLP-Movie_revenues_prediction(to do)"  # FILL IN THE ABSOLUTE PATH TO THE DATASET HERE
TRAIN_DIR = os.path.join(PATH_TO_DATA, "(to do)")
TEST_DIR = os.path.join(PATH_TO_DATA, "(to do)")


# to do
def nltk_tokenize_doc(doc):
    """
        Tokenize a document into bag-of-words tokens
    """
    bow = defaultdict(float)
    tokens = nltk.word_tokenize(doc.decode("utf8"))
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        if token not in stop:
            bow[token] += 1.0
    return bow


class NaiveBayes:
    def __init__(self):
        self.vocab = set()
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0 }
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0 }
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float) }

    def train_model(self, num_docs=None):
        if num_docs is not None:
            print "Limiting to only %s docs per clas" % num_docs

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)  # to do
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)  # to do
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p,f),'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        print "REPORTING CORPUS STATISTICS"
        print "NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL]
        print "NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL]
        print "NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL]
        print "NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL]
        print "VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab)

    def tokenize_and_update_model(self, doc, label):
        bow = nltk_tokenize_doc(doc)
        # self.update_model(bow, label)
        for (word, count) in bow.items():
            self.class_word_counts[label][word] += count
            self.class_total_word_counts[label] += count
            self.vocab.add(word)
        self.class_total_doc_counts[label] += 1

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        word_count = self.class_word_counts[label][word] + alpha
        label_word_count = self.class_total_word_counts[label] + alpha * len(self.vocab)
        return word_count / label_word_count

    def log_likelihood(self, bow, label, alpha):
        likelihood = 0
        for (word, count) in bow.items():
            likelihood += math.log(self.p_word_given_label_and_psuedocount(word, label, alpha))
        return likelihood

    def log_prior(self, label):
        doc = self.class_total_doc_counts[POS_LABEL] + self.class_total_doc_counts[NEG_LABEL]
        return math.log(self.class_total_doc_counts[label]) - math.log(doc)

    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        pos_prob = self.unnormalized_log_posterior(bow, POS_LABEL, alpha)
        neg_prob = self.unnormalized_log_posterior(bow, NEG_LABEL, alpha)
        if pos_prob > neg_prob:
            return POS_LABEL
        else:
            return NEG_LABEL

    def evaluate_classifier_accuracy(self, alpha):
        pos_path = os.path.join(TEST_DIR, POS_LABEL)
        neg_path = os.path.join(TEST_DIR, NEG_LABEL)
        count = 0
        total_doc = 0
        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            files = os.listdir(p)
            for file in files:
                with open(os.path.join(p, file), 'r') as doc:
                    content = doc.read()
                    total_doc += 1
                    if self.classify(nltk_tokenize_doc(content), alpha) == label:
                        count += 1
        return 100 * count / total_doc
# ----------------------------------------------------review done-------------------------------------------------------
if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model()
    print "accuracy(psuedocount:1) for test case: ", nb.evaluate_classifier_accuracy(1), "%"
    # nb.train_model(num_docs=10)
    # produce_hw1_results()