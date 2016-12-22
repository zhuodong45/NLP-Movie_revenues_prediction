from __future__ import division
import math
import os
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import string
import xml.etree.cElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


# tokenize for training data doc
def nltk_tokenize_doc(doc):
    bow = defaultdict(float)
    tokens = nltk.word_tokenize(doc.decode("utf8"))
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        if token not in stop:
            bow[token] += 1.0
    return bow

# tokenize for testing data review(string)
def nltk_tokenize_string(review):
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
        self.weekend_gross = np.array([])  # store weekend gross
        self.revenue = np.array([])  # store revenue
        self.review_recommend = np.array([])    # store review recommend

    # read in training data for processing
    def train_model(self, num_docs=None):
        if num_docs is not None:
            print "Limiting to only %s docs per clas" % num_docs
        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        for (p, label) in [(pos_path, POS_LABEL), (neg_path, NEG_LABEL)]:
            filenames = os.listdir(p)
            if num_docs is not None: filenames = filenames[:num_docs]
            for f in filenames:
                with open(os.path.join(p, f), 'r') as doc:
                    content = doc.read()
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def report_statistics_after_training(self):
        print "Review classifier training data"
        print "Number of document in positive class:", self.class_total_doc_counts[POS_LABEL]
        print "Number of document in negative class:", self.class_total_doc_counts[NEG_LABEL]
        print "Number of tokens in positive class:", self.class_total_word_counts[POS_LABEL]
        print "Number of tokens in negative class:", self.class_total_word_counts[NEG_LABEL]
        print "Vocabulary size:", len(self.vocab)

    # update training data information
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

    # read in testing data and predict revenue
    def testing_process(self, alpha):
        total_review = 0
        total_pos_review = 0
        total_neg_review = 0
        files = os.listdir(TEST_DIR)
        for file in files:
            pos_count = 0
            doc_review = 0
            doc = os.path.join(TEST_DIR, file)
            root = ET.parse(doc)

            # update revenue list
            us_gross = root.find('.//US_Gross')
            us_gross = float(us_gross.text.replace('$', '').replace(',', ''))
            if us_gross == 0:
                continue
            self.revenue = np.append(self.revenue, [us_gross])

            # update weekend gross list
            gross = root.find('.//weekend_gross')
            gross = float(gross.text.replace('$', '').replace(',', ''))
            self.weekend_gross = np.append(self.weekend_gross, [gross])

            # update review recommend
            for review in root.findall('.//snippet'):
                doc_review += 1
                total_review += 1
                if self.classify(nltk_tokenize_string(review.text), alpha) == POS_LABEL:
                    pos_count += 1
                    total_pos_review += 1
                else:
                    total_neg_review += 1
            recommend = 100 * pos_count / doc_review
            self.review_recommend = np.append(self.review_recommend, [recommend])

        print "Total review count: ", total_review
        print "Positive review count: ", total_pos_review
        print "negative review count: ", total_neg_review
        print "First 10 recommend examples: ", self.review_recommend[:10]
        print "First 10 revenue examples: ", self.revenue[:10]
        print "First 10 weekend gross examples: ", self.weekend_gross[:10]

        # split data into linear regression training set and testing set
        split_list = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
        self.review_recommend = split_list(self.review_recommend, 1000)
        self.weekend_gross = split_list(self.weekend_gross, 1000)
        self.revenue = split_list(self.revenue, 1000)
        print "Linear regression training data size: ", len(self.revenue[0])
        print "Linear regression testing data size: ", len(self.revenue[1])

        # linear regression with weekend gross and revenue
        p = np.polyfit(self.weekend_gross[0], self.revenue[0], 1)   # calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.weekend_gross[0], self.revenue[0])
        print "Linear regression coefficients: ", slope
        print "Linear regression intercept: ", intercept
        print "R-square: ", r_value**2
        m1 = p[0]
        c1 = p[1]
        print "Linear regression function: ", "y =", m1, "* weekend gross +", c1

        plt.ylim([0, 350000000])
        plt.xlim([0, 80000000])
        plt.plot(self.weekend_gross[0], self.revenue[0], 'o')   # plot training data points in the graph
        plt.plot(self.weekend_gross[0], np.polyval(p, self.weekend_gross[0]), 'r-')  # plot line in the graph
        plt.show()  # display plot

        plt.ylim([0, 350000000])
        plt.xlim([0, 80000000])
        plt.plot(self.weekend_gross[1], self.revenue[1], 'o')  # plot test data points in the graph
        plt.plot(self.weekend_gross[0], np.polyval(p, self.weekend_gross[0]), 'r-')  # plot line in the graph
        plt.show()

        test_weekend_gross = self.weekend_gross[1]
        test_revenue = self.revenue[1]
        bias_list = []

        size = len(test_weekend_gross)
        success_count = 0
        for i in range(size):
            predict = m1 * test_weekend_gross[i] + c1   # predict result
            bias = predict / test_revenue[i]    # bias with real revenue
            if 0.6 < bias < 1.4:    # success range
                success_count += 1
            bias_list.append(bias)
        print "Success rate: ", success_count/size  # success percentage

        # linear regression with weekend gross, review and revenue
        x1 = self.review_recommend[0]
        x2 = self.weekend_gross[0]
        y = self.revenue[0]
        x = np.transpose(np.array([x1, x2]))    # 2D array
        m2, c2 = np.linalg.lstsq(x, y)[0]   # calculate linear regression with multiple variables
        plt.ylim([0, 400000000])
        plt.xlim([0, 80000000])
        plt.plot(x, y, 'o', label='Original data')
        plt.plot(x, m2 * x + c2, 'r', label='Fitted line')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model()
    nb.testing_process(1)