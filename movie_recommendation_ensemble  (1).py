#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd


# In[89]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[90]:


movies=movies.merge(credits,on='title')


# In[91]:


movies.head(3)


# In[92]:


#genres
#id
#keywords
#title
#overview
#cast
#crew
#ignoring the numerical values, and taking the most import columns into consideration.


# In[93]:


movies['original_language'].value_counts()


# In[94]:


movies.info()


# In[95]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[96]:


movies.head(3)


# In[97]:


#checking for the null values in the columns
#Removing the rows which contains the null values


# In[98]:


movies.isnull().sum()


# In[99]:


movies.dropna(inplace=True)


# In[100]:


#checking for the duplicate words


# In[101]:


movies.duplicated().sum()


# In[102]:


movies.iloc[0].genres


# In[103]:


#As we are having the list of dictionaries which is complex and the each dictionary contains id which is numerical value
#Which is not required so we are simply exacting the required words into list
#for genre we need only the type example adventure, romace etc...
#so we are taking the function and creating the list to get the words into list but here the list is of the string which is not
#iteratable so we are using ast library to evaluate the string literals
#using the apply function to extract the genre keywords into list of all rows in the column
#we are applying for {genre,keywords}


# In[104]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[105]:


movies['genres']=movies['genres'].apply(convert)


# In[106]:


movies.head()


# In[107]:


movies['keywords']=movies['keywords'].apply(convert)


# In[108]:


movies.head()


# In[109]:


#the cast contains many personalities but the for our model we are required of top three names, because the people search
#based on the hero, heroine, topcast(main side character, side herine etc.. famous) so we are taking the topmost three values.


# In[110]:


import ast
def convert3(obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            L.append(i['name'])
        else:
            break
    return L


# In[111]:


movies['cast']=movies['cast'].apply(convert3)


# In[112]:


movies.head()


# In[113]:


#for crew we are seraching for the name director in the list of dictionary and extracting the director name.
#as people search the movie via director too.


# In[114]:


def fetch_dirc(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break;
    return L


# In[115]:


movies['crew']=movies['crew'].apply(fetch_dirc)


# In[116]:


movies.head()


# In[117]:


#converting the overview column into list for further concatenation of the five columns into one column.


# In[118]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[119]:


movies.head()


# In[120]:


#replacing spaces between the words, for instance our model given a task to find the movies of the director Sam Worthington
# as the model moves from word to word so while giving suggestion of the movies the model might give the movies which were
# directed by other directors whose name starts with Sam


# In[121]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[122]:


movies.head()


# In[123]:


#concatination of overview,genres,keywords,cast,crew into single section


# In[124]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[125]:


movies.head()


# In[126]:


#creating new dataframe for the three column sheet


# In[127]:


new_data=movies[['movie_id','title','tags']]


# In[128]:


new_data.head()


# In[129]:


#conversion of whole list into Strings


# In[130]:


new_data['tags']=new_data['tags'].apply(lambda x:" ".join(x))


# In[131]:


new_data.head()


# In[132]:


new_data['tags']=new_data['tags'].apply(lambda x:x.lower())


# In[133]:


new_data.head()


# In[134]:


#we are using nltk library to stem the words ex if we have words such as play,played,playing using stem we can reduce them
#to single word because we are extractig features the we are taking most repeated words as play,played,playing have same
#significance we are taking them as on so we are taking whole tags section and doing the stem function


# In[135]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[136]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    string=" ".join(y) 
    return string


# In[137]:


#sample example of stem
ps.stem('playing')
ps.stem('play')
ps.stem('played')


# In[138]:


new_data['tags'][0]


# In[139]:


# another example
stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver stephenlang michellerodriguez giovanniribisi joeldavidmoore cchpounder wesstudi lazalonso dileeprao mattgerald seananthonymoran jasonwhyte scottlawrence kellykilgour jamespatrickpitt seanpatrickmurphy peterdillon kevindorman kelsonhenderson davidvanhorn jacobtomuri michaelblain-rozgay joncurry lukehawker woodyschultz petermensah soniayee jahnelcurfman ilramchoi kylawarren lisaroumain debrawilson chrismala taylorkibby jodielandau julielamm cullenb.madden josephbradymadden frankietorres austinwilson sarawilson tamicawashington-miller lucybriant nathanmeister gerryblair matthewchamberlain paulyates wraywilson jamesgaylyn melvinlenoclarkiii carvonfutrell brandonjelkes micahmoch hanniyahmuhammad christophernolen christaoliver aprilmariethomas bravitaa.threatt colinbleasdale mikebodnar mattclayton nicoledionne jamieharrison allanhenry anthonyingruber ashleyjeffery deanknowsley josephmika-hunt terrynotary kaipantano loganpithyou stuartpollock raja garethruck rhiansheehan t.j.storm jodietaylor aliciavela-bailey richardwhiteside nikiezambo julenerenee jamescameron')


# In[140]:


new_data['tags']=new_data['tags'].apply(stem)


# In[141]:


#we are vectorising the content using the bag of words technique to extract the most important words for the model
#the most repeated words in whole tags rows and they are taken as features we are taking max features as 5000
# we are using sklearn library for the vectorisation


# In[142]:


from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000, stop_words='english')


# In[143]:


vectors=cv.fit_transform(new_data['tags']).toarray()


# In[144]:


vectors[0]


# In[145]:


cv.get_feature_names_out()


# In[146]:


cv.get_feature_names_out().shape


# In[147]:


#now we are calcualting the distance between each row in high dimenstional space using cosine similarty using this we can easily
# calculate the distance using angles, low angle between the row means they are more similar, if the high angle - not similar
#ex angle between two vectors is 5 degree then they are more similar compared to the ones with 180 degrees


# In[148]:


from sklearn.metrics.pairwise import cosine_similarity


# In[149]:


similarity=cosine_similarity(vectors)


# In[150]:


#we are taking distance between the each movie to all the remaining movies.


# In[151]:


similarity.shape


# In[152]:


#this array contains arrays where each movie array contains the distance between the other movies.


# In[153]:


similarity[0]


# In[154]:


# we are creating the main function here where the real recommendation works,here we are taking only five similar movies
#our vector array i.e the similarity array contains the similar movies percentage to others,but to get the top 5 similarity
# we need to sort the array but if sort the array the index of the movies are jumbles so we are using enumerate function
# to label each the index postion so we can retrive the movies using index position
# we are using reverse to get decending order of the recommendations for more similarity,using lambda to sort according
# to percentage value only, not with the indexes we labelled for each percentage for similarity(index=movie position)
# at last we print the title names using the index postions of the movies 


# In[155]:


def recommend(movie):
    movie_index=new_data[new_data['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_data.iloc[i[0]].title)


# In[156]:


recommend('Batman')


# In[157]:


new_data['title'].values


# In[158]:


# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure that all entries in 'tags' are strings by joining lists for TF-IDF
# new_data['tags_str'] = new_data['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# Now, re-run the TF-IDF vectorizer on the 'tags' column
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_data['tags'])

# Calculate Cosine Similarity Matrix
cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on TF-IDF and cosine similarity
def get_recommendations_tfidf(title, cosine_sim=cosine_sim_tfidf):
    idx = new_data[new_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return new_data['title'].iloc[movie_indices]

# Example usage
get_recommendations_tfidf('The Dark Knight')


# In[159]:


from sklearn.decomposition import TruncatedSVD

# Apply SVD to reduce dimensionality of the TF-IDF matrix
svd = TruncatedSVD(n_components=100, random_state=42)
lsi_matrix = svd.fit_transform(tfidf_matrix)

# Calculate Cosine Similarity in the reduced LSI space
cosine_sim_lsi = cosine_similarity(lsi_matrix)

# Function to get movie recommendations using LSI
def get_recommendations_lsi(title, cosine_sim=cosine_sim_lsi):
    idx = new_data[new_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example: Get LSI-based recommendations
get_recommendations_lsi('The Dark Knight')


# In[160]:


get_ipython().system('pip install rank-bm25')


# In[161]:


from rank_bm25 import BM25Okapi

# Assuming each entry in movies['tags'] is already a list of tags
tokenized_tags = [tags for tags in new_data['tags']]

# Build BM25 model
bm25 = BM25Okapi(tokenized_tags)

#for streamlit purpose
bm25streamlit=tokenized_tags

# Function to get BM25-based recommendations
def get_recommendations_bm25(title):
    idx = new_data[new_data['title'] == title].index[0]
    query = new_data['tags'].iloc[idx]  # Use the list directly
    scores = bm25.get_scores(query)
    sim_scores = list(enumerate(scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return new_data['title'].iloc[movie_indices]

# Example: Get BM25-based recommendations
print(get_recommendations_bm25('The Dark Knight'))


# In[162]:


import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Create a new column for tokenized tags
new_data['tokenized_tags'] = new_data['tags'].apply(lambda x: x.split())  # Tokenize the 'tags' column

# Train Word2Vec model using the tokenized tags
word2vec = Word2Vec(sentences=new_data['tokenized_tags'], vector_size=100, window=5, min_count=1, workers=4)

# Create a movie vector by averaging word vectors from the tokenized tags
def get_movie_vector(movie_idx):
    words = new_data['tokenized_tags'].iloc[movie_idx]  # Use the new tokenized_tags column
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec.vector_size)

# Calculate similarity between movie vectors
movie_vectors = np.array([get_movie_vector(idx) for idx in range(len(new_data))])
cosine_sim_word2vec = cosine_similarity(movie_vectors)

# Function to get Word2Vec-based recommendations
def get_recommendations_word2vec(title):
    idx = new_data[new_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim_word2vec[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return new_data['title'].iloc[movie_indices]

# Example: Get Word2Vec-based recommendations
recommendations = get_recommendations_word2vec('The Dark Knight')
print(recommendations)


# In[163]:


import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Tokenize the 'tags' column into individual words
new_data['tokenized_tags'] = new_data['tags'].apply(lambda x: x.split())

# Use MultiLabelBinarizer to transform tags into binary vectors
mlb = MultiLabelBinarizer()
binary_vectors = mlb.fit_transform(new_data['tokenized_tags'])

# Function to get Jaccard similarity between two movies
def get_jaccard_similarity(movie_idx1, movie_idx2):
    binary_vector1 = binary_vectors[movie_idx1]
    binary_vector2 = binary_vectors[movie_idx2]
    
    # Compute Jaccard similarity
    return jaccard_score(binary_vector1, binary_vector2)

# Function to get Jaccard-based recommendations
def get_recommendations_jaccard(title):
    idx = new_data[new_data['title'] == title].index[0]
    sim_scores = [(i, get_jaccard_similarity(idx, i)) for i in range(len(new_data))]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return new_data['title'].iloc[movie_indices]

# Example: Get Jaccard-based recommendations
recommendations = get_recommendations_jaccard('The Dark Knight')
print(recommendations)


# In[164]:


from collections import Counter

# Ensemble function to get majority voting across different algorithms
def get_ensemble_recommendations(title):
    recommendations = []
    
    # Get recommendations from each model
    recommendations += list(get_recommendations_tfidf(title))
    recommendations += list(get_recommendations_lsi(title))
    recommendations += list(get_recommendations_bm25(title))
    recommendations += list(get_recommendations_word2vec(title))
    recommendations += list(get_recommendations_jaccard(title))
    
    # Use majority voting
    top_recommendations = Counter(recommendations).most_common(5)
    return [rec[0] for rec in top_recommendations]

# Example: Get ensemble-based recommendations
get_ensemble_recommendations("The Avengers")


# In[ ]:


import pickle
pickle.dump(new_data, open('../../Downloads/movies.pkl', 'wb'))


# In[146]:


pickle.dump(new_data.to_dict(),open('movie_dict.pkl','wb'))


# In[147]:


pickle.dump(similarity, open('../../Downloads/similarity.pkl', 'wb'))


# In[148]:


pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(cosine_sim_tfidf, open('cosine_sim_tfidf.pkl', 'wb'))
pickle.dump(cosine_sim_lsi, open('cosine_sim_lsi.pkl', 'wb'))
pickle.dump(cosine_sim_word2vec, open('cosine_sim_word2vec.pkl', 'wb'))
pickle.dump(binary_vectors, open('binary_vectors.pkl', 'wb'))


# In[168]:


import pickle
pickle.dump(bm25streamlit, open('bm25_corpus.pkl', 'wb'))


# In[ ]:




