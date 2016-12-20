from __future__ import division
import math
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import string
import xml.etree.cElementTree as ET

# -------------------------------------------classifier for review------------------------------------------------------
# stopwords and punctuation
stop = set(stopwords.words()).union(set(string.punctuation))
# remove too common words
stop.update(['the', 'and', 'a', 'of', 'to', 'is', 'in', 'i', 'it', 'that'])  # to do

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA_train = r"E:\cs585project\hw3\hw3_large_movie_review_dataset"
TEST_DIR = r"E:\cs585project\NLP-Movie_revenues_prediction\movies-data-v1.0\metacritic+starpower+holiday+revenue+screens+reviews"
TRAIN_DIR = os.path.join(PATH_TO_DATA_train, "train")


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


def nltk_tokenize_string(review):
    """
        Tokenize a document into bag-of-words tokens
    """
    bow = defaultdict(float)
    tokens = review.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        if token not in stop:
            bow[token] += 1.0
    return bow


class NaiveBayes:
    def __init__(self):
        self.vocab = set()
        self.class_total_doc_counts = {POS_LABEL: 0.0,
                                       NEG_LABEL: 0.0}
        self.class_total_word_counts = {POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0}
        self.class_word_counts = {POS_LABEL: defaultdict(float),
                                  NEG_LABEL: defaultdict(float)}
        self.weekend_gross = []
        self.revenue = []
        self.review_recommend = []

    def train_model(self, num_docs=None):
        if num_docs is not None:
            print "Limiting to only %s docs per clas" % num_docs

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        print "Starting training with paths %s and %s" % (pos_path, neg_path)
        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p, f), 'r') as doc:
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

    def read_dataset(self, alpha):
        # for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
        files = os.listdir(TEST_DIR)
        i = 0
        for file in files:
            pos_count = 0
            total_review = 0
            doc = os.path.join(TEST_DIR, file)
            root = ET.parse(doc)
            for review in root.findall('.//snippet'):
                total_review += 1
                if self.classify(nltk_tokenize_string(review.text), alpha) == POS_LABEL:
                    pos_count += 1
            recommend = 100 * pos_count / total_review
            self.review_recommend.append(recommend)
            gross = root.find('.//weekend_gross')
            self.weekend_gross.append(gross)
            us_gross = root.find('.//US_Gross')
            self.revenue.append(us_gross)
        split_list = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
        self.review_recommend = split_list(self.review_recommend, 1000)
        self.weekend_gross = split_list(self.weekend_gross, 1000)
        self.revenue = split_list(self.revenue, 1000)


# ----------------------------------------------------review done-------------------------------------------------------
if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model()
    nb.read_dataset(1)
    # nb.train_model(num_docs=10)
    # produce_hw1_results()
