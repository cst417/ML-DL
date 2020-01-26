import pandas as pd
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting= 3)

corpus =[]
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', data['Review'][i]) #keeps a-z and A-Z, replaces useless stuff with space.
    review = review.lower() 
    review = review.split() #easier to clean when separated 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)

print(corpus)
# # Cleaning the texts for 1 review
# review = re.sub('[^a-zA-Z]',' ', data['Review'][0]) #keeps a-z and A-Z, replaces useless stuff with space.
# review = review.lower() 
# # print(review)
# # Get rid of irrelevant words
# # nltk.download('stopwords')
# #split the review into separate words
# review = review.split() #easier to clean when separated 
# ps = PorterStemmer()
# review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  #set makes it faster, ps.stem stemms every word and puts them in the review

# # reverting review back to a sentence
# review = ' '.join(review)
# print(review)





