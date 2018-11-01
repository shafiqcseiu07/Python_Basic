import nltk
nltk.download('stopwords')
from nltk.corpus import movie_reviews as mr
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
stop = stopwords.words('english')

all_words = FreqDist(w.lower() for w in mr.words() if w.lower() not in stop and w.lower() not in string.punctuation)

print (all_words)
