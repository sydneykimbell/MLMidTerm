import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.util import ngrams
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import contractions
import regex as re
import string
import pylev
import warnings

warnings.filterwarnings(action='ignore')

# RESOURCES REFERENCED: 
# techniques for text preprocessing in NLP: https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/
# expand contractions: https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/
# http://www.kozareva.com/papers/fintalKozareva.pdf
# cosine similarity: https://www.machinelearningplus.com/nlp/cosine-similarity/
# https://link.springer.com/chapter/10.1007/978-3-319-59692-1_14
# bleu score: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# headers for dataframes
header_train = ["train_id", "sentence_1", "sentence_2", "output"]
header_dev = ["dev_id", "sentence_1", "sentence_2", "output"]
header_test = ["test_id", "sentence_1", "sentence_2"]

# read the data into dataframes

df_train = pd.read_csv('train_with_label.txt', sep='\t+', on_bad_lines='skip',  names=header_train, engine='python').dropna()
df_dev = pd.read_csv('dev_with_label.txt', sep='\t+', on_bad_lines='skip', names=header_dev, engine='python').dropna()
df_test = pd.read_csv('test_without_label.txt', sep='\t+', on_bad_lines='skip', names=header_test, engine='python').dropna()


def fix_punctuation(sentence):
    '''removes the space before punctuation'''
    sentence = re.sub(r'\s([,?.!"](?:\s|$))', r'\1', sentence)
    sentence = re.sub(r"\b\s+'\b", r"'", sentence)
    return sentence

def expand_contractions(sentence):
    '''expands contractions'''
    expanded_words = []
    for word in sentence.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    return expanded_text

def lowercase(sentence):
    '''make every character in the sentence lowercase'''
    sentence = sentence.lower()
    return sentence

def remove_punctuation(sentence):
    '''remove all punctuation from the sentence'''
    sentence = re.sub('[%s]' % re.escape(string.punctuation), '' , sentence)
    return sentence

def remove_digits(sentence):
    '''remove all words and digits containing digits from the sentence'''
    sentence = ' '.join(s for s in sentence.split() if not any(c.isdigit() for c in s))
    return sentence

def remove_stopwords(sentence):
    '''remove all stop words from sentence'''
    stop_words = set(stopwords.words('english'))
    sentence = " ".join([word for word in str(sentence).split() if word not in stop_words])
    return sentence

def lemmatization(sentence):
    '''lemmatize sentence'''
    lemmatizer = WordNetLemmatizer()
    sentence = " ".join([lemmatizer.lemmatize(word) for word in sentence.split()])
    return sentence

def stem(sentence):
    '''stem sentence'''
    stemmer = PorterStemmer()
    sentence = " ".join([stemmer.stem(word) for word in sentence.split()])
    return sentence


def remove_white_space(sentence):
    '''remove white space from sentence'''
    sentence = re.sub(' +', ' ', sentence)
    return sentence

def preprocess(df):
    df['sentence_1'] = df['sentence_1'].apply(lambda x: fix_punctuation(x))
    df['sentence_1'] = df['sentence_1'].apply(lambda x: expand_contractions(x))
    df['sentence_1'] = df['sentence_1'].apply(lambda x: lowercase(x))
    df['sentence_1'] = df['sentence_1'].apply(lambda x: remove_punctuation(x))
    #df['sentence_1'] = df['sentence_1'].apply(lambda x: remove_stopwords(x))
    df['sentence_1'] = df['sentence_1'].apply(lambda x: lemmatization(x))
    df['sentence_1'] = df['sentence_1'].apply(lambda x: remove_white_space(x))
    df['sentence_1'] = df['sentence_1'].apply(lambda x: x.split())

    df['sentence_2'] = df['sentence_2'].apply(lambda x: fix_punctuation(x))
    df['sentence_2'] = df['sentence_2'].apply(lambda x: expand_contractions(x))
    df['sentence_2'] = df['sentence_2'].apply(lambda x: lowercase(x))
    df['sentence_2'] = df['sentence_2'].apply(lambda x: remove_punctuation(x))
    #df['sentence_2'] = df['sentence_2'].apply(lambda x: remove_stopwords(x))
    df['sentence_2'] = df['sentence_2'].apply(lambda x: lemmatization(x))
    df['sentence_2'] = df['sentence_2'].apply(lambda x: remove_white_space(x))
    df['sentence_2'] = df['sentence_2'].apply(lambda x: x.split())
    return df

def lengths(df):
    '''get the lengths of sentence 1 and sentence 2'''
    lengths = pd.DataFrame(columns=['length_s1', 'length_s2'])
    length_s1 = []
    length_s2 = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for sentence in sentence_1:
        length_s1.append(len(sentence))
    for sentence in sentence_2:
        length_s2.append(len(sentence))
    
    lengths['length_s1'] = length_s1
    lengths['length_s2'] = length_s2
    
    return lengths

def length_difference(df):
    '''absolute value of the difference between the lengths of the two sentences'''
    feature = pd.DataFrame(columns=['length_difference'])
    length_difference = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for s1, s2 in zip(sentence_1, sentence_2):
        diff = (abs(len(s1)-len(s2)))
        length_difference.append(diff)
    feature['length_difference'] = length_difference

    return feature

def overlap(df):
    '''find a proportion of overlapping words'''
    feature = pd.DataFrame(columns=['overlap'])
    overlap = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for s1, s2 in zip(sentence_1, sentence_2):
        combined = s1 + s2
        unique = len(set(combined))
        common = len(list(set(s1) & set(s2)))
        overlap.append(common / unique)
    feature['overlap'] = overlap

    return feature

def sentence_length_difference(df):
    '''find a proportion of sentence length difference'''
    feature = pd.DataFrame(columns=['SLD'])
    sld = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for s1, s2 in zip(sentence_1, sentence_2):
        sld.append((len(s1) - len(s2))/(len(s1)))
    feature['SLD'] = sld

    return feature

def sentence_length_difference2(df):
    '''alternate way of finding sentence length difference'''
    feature = pd.DataFrame(columns=['SLD*'])
    sld = []
    d = 0.8
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for s1, s2 in zip(sentence_1, sentence_2):
        sld.append(1/(d**(len(s1)-len(s2))))
    feature['SLD*'] = sld

    return feature

def levenshtein_distance(df):
    '''find levenshtein distance'''
    feature = pd.DataFrame(columns=['Levenshtein Distance'])
    lev_dist = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for s1, s2 in zip(sentence_1, sentence_2):
        lev_dist.append(pylev.levenshtein(s1,s2))
    feature['Levenshtein Distance'] = lev_dist

    return feature

def cosine_similarity(df):
    '''vectorize sentences and find cosine similarity'''
    feature = pd.DataFrame(columns=['Cosine Similarity'])
    cos_sim = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    #sw = stopwords.words('english')
    for s1, s2 in zip(sentence_1, sentence_2):
        l1 = []
        l2 = []
        X_set = set(s1)
        Y_set = set(s2)
        #X_set = {w for w in s1 if not w in sw} 
        #Y_set = {w for w in s2 if not w in sw}

        rvector = X_set.union(Y_set)
        for w in rvector:
            if w in X_set:
                l1.append(1)
            else:
                l1.append(0)
            if w in Y_set:
                l2.append(1)
            else:
                l2.append(0)
        
        c = 0
        for i in range(len(rvector)):
            c += l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5)
        cos_sim.append(cosine)
    feature['Cosine Similarity'] = cos_sim

    return feature

def shared_words(df):
    '''number of shared words between the two sentences'''
    feature = pd.DataFrame(columns=['shared_words'])
    common = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    for s1, s2 in zip(sentence_1, sentence_2):
        s1 = set(s1)
        s2 = set(s2)
        common.append((len(s1.intersection(s2))))
    
    feature['shared_words'] = common
    return feature

def bleu_score(df):
    '''find bleu score for unigrams, bigrams, and trigrams'''
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    features = pd.DataFrame(columns=["bleu_1", "bleu_2", "bleu_3"])
    bleu1 = []
    bleu2 = []
    bleu3 = []

    for s1,s2 in zip(sentence_1, sentence_2):

        bleu1.append(nltk.translate.bleu_score.sentence_bleu([s1], s2, weights=[1]))
        bleu2.append(nltk.translate.bleu_score.sentence_bleu([s1], s2, weights=[0.5, 0.5]))
        bleu3.append(nltk.translate.bleu_score.sentence_bleu([s1], s2, weights=[1/3, 1/3, 1/3]))

    features["bleu_1"] = bleu1
    features["bleu_2"] = bleu2
    features["bleu_3"] = bleu3

    return features

def meteor_scores(df):
    '''find meteor score'''
    features = pd.DataFrame(columns=["meteor_score"])
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values

    meteor_score = []
    for s1, s2 in zip(sentence_1, sentence_2):
        meteor_score.append(nltk.translate.meteor_score.single_meteor_score(s1, s2))
    features["meteor_score"] = meteor_score
    return features

def jaccard_similarity(df):
    '''find jaccard similarity'''
    features = pd.DataFrame(columns=["jaccard_similarity"])
    jac_sim = []
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values

    for s1, s2 in zip(sentence_1, sentence_2):
        intersection_cardinality = len(set.intersection(*[set(s1), set(s2)]))
        union_cardinality = len(set.union(*[set(s1), set(s2)]))
        jac_sim.append(intersection_cardinality/float(union_cardinality))
    
    features['jaccard_similarity'] = jac_sim
    return features

def nist_score(df):
    '''find nist score for n=1'''
    sentence_1 = df['sentence_1'].values
    sentence_2 = df['sentence_2'].values
    features = pd.DataFrame(columns=["nist_score"])
    nist = []

    for s1, s2 in zip(sentence_1, sentence_2):
        nist.append((nltk.translate.nist_score.sentence_nist([s1], s2, n=1)))

    features["nist_score"] = nist

    return features



def features(df):
    '''select features to use in the model'''
    lens = lengths(df)
    diff = length_difference(df)
    length_diff = sentence_length_difference(df)
    length_diff2 = sentence_length_difference2(df)
    common = shared_words(df)
    over = overlap(df)
    lev_dist = levenshtein_distance(df)
    cos_sim = cosine_similarity(df)
    bleu = bleu_score(df)
    ms = meteor_scores(df)
    js = jaccard_similarity(df)
    nist = nist_score(df)
    res = [bleu, over, cos_sim, lev_dist, diff, ms, nist]
    #res = [cos_sim, lev_dist, diff, common, nist]
    features = pd.concat(res, axis=1)
    return features


# preprocess all of the data

df_train = preprocess(df_train)
df_dev = preprocess(df_dev)
df_test = preprocess(df_test)

# extract the features

X_train = features(df_train)
X_dev = features(df_dev)
X_test = features(df_test)

# get test_id column to write the output file

X_test_id = df_test['test_id'].values

# turn the dataframes into arrays to fit the model

X_train = X_train.values
X_dev = X_dev.values
X_test = X_test.values
Y_train = df_train['output'].values
Y_dev = df_dev['output'].values

# make classifier and fit to the training set

classifier = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', gamma=0.9, C=0.4, class_weight='balanced'))
classifier.fit(X_train, Y_train)

# get predictions on test set

Y_pred = classifier.predict(X_test)

# accuracy score on the dev set

print(classifier.score(X_dev, Y_dev))

# write the results of the test set to a new file

file = open('SydneyKimbell_test_result.txt', 'w')

for i in range(len(X_test_id)):
    file.write(X_test_id[i] + '\t' + str(Y_pred[i]) + '\n')


