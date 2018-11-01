import nltk
import string
#from nltk import string
#nltk.download('string')

#nltk.download('gutenberg')
#nltk.corpus.gutenberg.files()
from nltk.corpus import movie_reviews

#reviews = CategorizedPlaintextCorpusReader('./nltk_data/corpora/movie_reviews', r'(\w+)/*.txt', cat_pattern=r'/(\w+)/.txt')
#location of movie reviews C:\Users\SPINT-13\AppData\Roaming\nltk_data\corpora\movie_reviews but u hv to use / instead of \ o directory

reviews = CategorizedPlaintextCorpusReader('./Roaming/nltk_data/corpora/movie_reviews', r'.*/.txt', cat_pattern=r'\d+_(\w+)/.txt')
reviews.categories()
# ['pos', 'neg']

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

all_words=nltk.FreqDist(
    w.lower()
    for w in movie_reviews.words()
    if w.lower() not in nltk.corpus.stopwords.words('english') and w.lower() not in  string.punctuation)
word_features = all_words.keys()[:100]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
print (document_features(movie_reviews.words('pos/11.txt')))

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print (nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
