# Standard libraries for data frames, arrays, and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

# Import Scikit Learn for preprocessing tools and algorithms
import sklearn

# Import library for setting dates
import datetime as dt

# Import library for reading pdf files
import PyPDF2

# Import wordclouds for visualizing topics
from wordcloud import WordCloud

# Import gensim library for calculating coherence measures
import gensim

# Import natural language package to remove stopwords and lemmatize terms
import nltk

# Import digits module to filter out numbers
import string

# Import statsmodels for statistical analysis
import statsmodels.api as sm

# Import pandas_datareader to get financial data from Yahoo
import pandas_datareader as pdr
from PyPDF2 import PdfReader

from matplotlib.dates import date2num

import yfinance as yf

# Import in unprocessed statements scraped from Scrapy into a csv file
data = pd.read_csv('statements.csv', skipinitialspace=True)
#print(data)
statements = data['Statement']

"""
The text preprocessing steps
The preprocessing steps I take here are:

Lowercasing
Remove extra spacing
Remove punctuation
Removing digits
Remove stopwords
Lemmatization
"""

# Import tools to tokenize, remove stopwords, and lemmatize (or stem) words
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords') # download collection of stopwords from NLTK
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
nltk.download('wordnet') # download wordnet dictionary of lemmas
from nltk.stem import WordNetLemmatizer
from string import digits

# Lowercase, remove digits, and lemmatize statements
n=len(statements)
raw_doc_length = list(range(0, n))
doc_length = list(range(0, n))
#porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
for i in range(n):
    # Lowercase the statement
    lowercase = statements[i].lower()
    # Tokenize the statement
    tokenize = CountVectorizer().build_tokenizer()(lowercase)
    # Count raw number of words
    raw_doc_length[i] = len(tokenize)
    # Remove digits
    remove_digits = [w for w in tokenize if not w.isdigit()]
    # Remove stopwords
    no_stopwords = [w for w in remove_digits if not w in stopwords.words('english')]
    # Lemmatize
    lemma = [lemmatizer.lemmatize(w) for w in no_stopwords]
    # Count preprocessed words
    doc_length[i] = len(lemma)
    # Join each statement back together
    statements[i] = ' '.join(lemma)

#print(statements[0])

# Save dates in a list
dates=data['Date']
x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in dates]

# Transform raw document length list into a numpy array
raw_doc_length = np.asarray(raw_doc_length)

# Transform preprocessed document length list into a numpy array
doc_length = np.asarray(doc_length)

# Create plot of document length
data = pd.DataFrame({'Year': x, 'Raw':raw_doc_length, 'Preprocessed':doc_length})
ax1=sns.lineplot(x='Year', y='Raw', data=data)
ax2=sns.lineplot(x='Year', y='Preprocessed', data=data)
ax1.set(xlabel ='', ylabel='Word Count')
ax2.set(xlabel ='', ylabel='Word Count')
ax1.text(x=date2num(data["Year"].iloc[-1])+30, y=data['Raw'].iloc[-1], s="Raw",
         horizontalalignment='left', size='small', verticalalignment='center')
ax2.text(x=date2num(data["Year"].iloc[-1])+30, y=data['Preprocessed'].iloc[-1], s="Preprocessed",
         horizontalalignment='left', size='small', verticalalignment='center')
# Shade in early 2000s crisis
ax2.axvspan(x[7], x[32], alpha=0.30, color='gray')
# Shade in global financial crisis
ax2.axvspan(x[65], x[105], alpha=0.30, color='gray')
plt.xlim(xmax=date2num(data["Year"].iloc[-1])+1826)
#plt.savefig('images/document_length.png', dpi=300)
plt.show()

#Part 2: Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer
# Create unigram TF-IDF weighted matrix
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=0.05, max_df=0.95)
X_tfidf = tfidf_vectorizer.fit_transform(statements).toarray()
print(X_tfidf.shape)

#Part 3: Topic Modeling with Non-negative Matrix Factorization
# Import NMF from Scikit Learn and
from sklearn.decomposition import NMF
from sklearn import metrics

# Create a function that prints the top words of a topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

# Create a function that prints the topics as a list
def print_topics(model, feature_names, n_top_words, n_components):
    topics=list(range(n_components))
    for topic_idx, topic in enumerate(model.components_):
        topics[topic_idx]=[str(feature_names[i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topics

# Get the terms from the TF-IDF matrix
tfidf_feature_names_n1 = tfidf_vectorizer.get_feature_names_out()

# Estimate preliminary unigram NMF model with three topics
nmf = NMF(n_components=3, random_state=1,
          beta_loss='kullback-leibler', solver='mu',
          max_iter=10000).fit(X_tfidf) # minimizes using kullback-leibler with unigrams
print_top_words(nmf, tfidf_feature_names_n1, 15)

#Part 4: Parameter Selection for Non-negative Matrix Factorization

# Retrieve modules to set corpous and get coherence. Show logging details to see progress.
from gensim import corpora, models

# Tokenize each statement, add terms to dictionary, and build main corpus
for i in range(n):
    statements[i] = CountVectorizer().build_tokenizer()(statements[i])
dictionary=corpora.Dictionary(statements)
corpus=[dictionary.doc2bow(doc) for doc in statements]

# Parameter selection using intrinsic coherence measure, u_mass. This measure uses the main corpus, which has its issues according to XXX paper
end_k=30
coherencenmf=[]
for k in range(3,end_k+1):
    n_components=k
    nmf = NMF(n_components=n_components, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=10000).fit(X_tfidf)
    topics=print_topics(nmf, tfidf_feature_names_n1, 15, n_components)
    cm_NMF = models.coherencemodel.CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherencenmf.append(cm_NMF.get_coherence())
    print('k= %d' % k)
    print_top_words(nmf, tfidf_feature_names_n1, 15)

# Plot the coherence over the different 'k' selections
fig, ax = plt.subplots()
ax.plot(list(range(3, end_k+1)), coherencenmf, 'b+-', linewidth=2, label='UMass Coherency', alpha=0.5, markevery=1)
ax.set_xlabel('k')
ax.set_ylabel('Coherence')
ax.annotate('Max TC-LCP @ k=3',
             xy=(3, -0.53001054467878117),
             xycoords='data',
             xytext=(7, -0.54),
             arrowprops=dict(arrowstyle="simple"))
plt.show()

best_nmf = NMF(n_components=3, random_state=1, beta_loss='kullback-leibler', solver='mu', max_iter=10000).fit(X_tfidf)
print_top_words(best_nmf, tfidf_feature_names_n1, 15)
best_topics=print_topics(best_nmf, tfidf_feature_names_n1, 15, 3)

# Extract the weights for the model for later regression analysis.
W = best_nmf.fit_transform(X_tfidf) # topic-document weights
H = best_nmf.components_ # word-topic weights

# Dimensions of the W matrix
print(W.shape)
print(H.shape)

#Part 5: Visualization of the NMF Topics

# Create dates and NMF weight variables
nmf_theme1_weights = W[:,0]
nmf_theme2_weights = W[:,1]
nmf_theme3_weights = W[:,2]

# A function to plot the weights of the themes
def plot_theme_weights(weights, name, color):
    fig, ax = plt.subplots()
    ax.plot(x, weights, '{}-'.format(color), linewidth=2, label='topic 1', alpha=0.6)
    ax.set_ylabel('Weight')
    ax.axvspan(x[7], x[32], alpha=0.30, color='gray')
    ax.axvspan(x[65], x[105], alpha=0.30, color='gray')
    plt.show()

# Plot topic 1 weights as a time series
plot_theme_weights(weights=nmf_theme1_weights, name='NMFWeights1', color='r')

# Plot topic 2 weights as a time series
plot_theme_weights(weights=nmf_theme2_weights, name='NMFWeights2', color='b')

# Plot topic 3 weights as a time series
plot_theme_weights(weights=nmf_theme3_weights, name='NMFWeights3', color='g')

# Next, I create word clouds for the topics using the wordcloud package.
from wordcloud import WordCloud
def create_wordcloud(words, name):
    topic = WordCloud(ranks_only=True, max_font_size=40, background_color="white").generate(' '.join(words))
    plt.figure(figsize=(10, 6))
    plt.imshow(topic, interpolation="bilinear")
    plt.axis("off")
    plt.show()

create_wordcloud(words=best_topics[0], name='NMF Theme 1')
create_wordcloud(words=best_topics[1], name='NMF Theme 2')
create_wordcloud(words=best_topics[2], name='NMF Theme 3')

#Topic Modeling using Latent Dirichelet Allocation
# Import LDA model
from sklearn.decomposition import LatentDirichletAllocation
# Parameter selection using intrinsic coherence measure, u_mass
coherencelda=[]
end_k = 30
for k in range(3,end_k+1):
    n_components=k
    ldamodel = LatentDirichletAllocation(doc_topic_prior = 0.5, topic_word_prior = 0.025, n_components=n_components, max_iter=10, learning_method='batch', random_state=0).fit(X_tfidf)
    topics=print_topics(ldamodel, tfidf_feature_names_n1, 15, n_components)
    cm_LDA = models.coherencemodel.CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherencelda.append(cm_LDA.get_coherence())
    print('k= %d' % k)
    print_top_words(ldamodel, tfidf_feature_names_n1, 15)
print(coherencelda)

# Plot the coherence over the different 'k' selections
fig, ax = plt.subplots()
ax.plot(list(range(3, end_k+1)), coherencelda, 'b+-', linewidth=2, label='U_Mass Coherence', alpha=0.5, markevery=1)
ax.legend()
plt.show()

lda = LatentDirichletAllocation(doc_topic_prior = 0.5, topic_word_prior = 0.025, n_components=3, max_iter=10, learning_method='batch', random_state=0).fit(X_tfidf)
print_top_words(lda, tfidf_feature_names_n1, 15)
lda_topics=print_topics(lda, tfidf_feature_names_n1, 15, 3)

# Get probabilities from LDA model
P = lda.transform(X_tfidf) # Get a topic-document probability matrix
print(P)

# Create variables of probabilities
lda_theme1_probabilities = P[:,0]
lda_theme2_probabilities = P[:,1]
lda_theme3_probabilities = P[:,2]

# Plot topic 1 probabilities as a time series
plot_theme_weights(weights=lda_theme1_probabilities, name='LDAProbabilities1', color='b')

# Plot topic 2 probabilities as a time series
plot_theme_weights(weights=lda_theme2_probabilities, name='LDAProbabilities2', color='r')

# Plot topic 3 probabilities as a time series
plot_theme_weights(weights=lda_theme3_probabilities, name='LDAProbabilities3', color='g')

"""
# Save lda probabilities
P = pd.DataFrame(P)
P.to_csv("results/probabilities.csv")
"""

create_wordcloud(words=lda_topics[0], name='LDA Theme 1')
create_wordcloud(words=lda_topics[1], name='LDA Theme 2')
create_wordcloud(words=lda_topics[2], name='LDA Theme 3')

# Plot the UMass coherence measure for both LDA and NMF
fig, ax = plt.subplots()
ax.plot(list(range(3, 30+1)), coherencenmf, 'b+-', linewidth=2, label='NMF', alpha=0.5, markevery=1)
ax.plot(list(range(3, 30+1)), coherencelda, 'r+-', linewidth=2, label='LDA', alpha=0.5, markevery=1)
ax.legend()
ax.set_ylabel('Coherence')
ax.set_xlabel('k')
plt.show()

# Establishing Polarity in FOMC Statements
# Read in pdf file of the list of negative financial terms
pdfFileObj = open('LM_Negative.pdf', 'rb')
pdfReader = PyPDF2.PdfReader(pdfFileObj)
num_pages = len(pdfReader.pages)
print(f'The number of pages in the PDF file is: {num_pages}')

# Collect words from each page and lowercase them
negative = []
for pageNum in range(0, num_pages):
    pageObj = pdfReader.pages[pageNum]
    negative.append(CountVectorizer().build_tokenizer()(pageObj.extract_text().lower()))  # 修改了 extract_text
print(negative)

# delete the first six terms that are simply part of the list description
del negative[0][0:6]

# Calculate the sum of negative terms within each statement
neg_sum = list(range(0, len(statements)))
for i in range(0, len(statements)):
    neg_count = 0
    fomc = statements[i]
    for word in fomc:
        for neg in negative:
            if((word in set(neg)) == True):
                neg_count += 1
    neg_sum[i] = neg_count

# Transform negative sum list into a numpy array and calculate percentage of negativity
neg_sum = np.asarray(neg_sum)
negativity_proportion = neg_sum / doc_length

# Export the proportion of negativity to csv file
NegProp = pd.DataFrame(negativity_proportion)
NegProp.to_csv("negativity.csv")

# Calculating Uncertainty in FOMC Statements

# Read in pdf file of list of uncertain financial terms
pdfFileObj2 = open('LM_Uncertainty.pdf','rb')
pdfReader2 = PyPDF2.PdfReader(pdfFileObj2)
num_pages2 = len(pdfReader2.pages)  # 修改了 pdfReader.pages 为 pdfReader2.pages
print(f'The number of pages in the PDF file is: {num_pages2}')

# Collect the uncertainty terms on each page into a list
uncertainty=[]
for pageNum in range(0, num_pages2):
    pageObj2 = pdfReader2.pages[pageNum]
    uncertainty.append(CountVectorizer().build_tokenizer()(pageObj2.extract_text().lower()))  # 修改了 extract_text
print(uncertainty)

# Delete the first six terms that are simply part of the list description
del uncertainty[0][0:6]

# Calculate the sum of uncertain terms within each statement
uncertain_sum = list(range(0, len(statements)))
for i in range(0, len(statements)):
    uncertain_count = 0
    fomc = statements[i]
    for word in fomc:
        for unc in uncertainty:
            if((word in set(unc)) == True):
                uncertain_count += 1
    uncertain_sum[i] = uncertain_count

# Transform the uncertainty sum list into a numpy array and calculate percentage of uncertainty
uncertain_sum = np.asarray(uncertain_sum)
uncertainty_proportion = uncertain_sum / doc_length

# Plot the proportion of uncertainty as a time series
fig, ax = plt.subplots()
ax.plot(x, uncertainty_proportion, 'b-', linewidth=2, label='Uncertainty', alpha=0.5, markevery=1)
ax.set_xlabel('year')
ax.set_ylabel('proportion of uncertainty')
plt.show()

# Export the uncertainty proportion to a csv file
# UncertProp = pd.DataFrame(uncertainty_proportion)
# UncertProp.to_csv("uncertainty.csv")

from pandas_datareader import data
import statsmodels.api as sm

# Want to look at influence of changes in topics
W_diff = np.diff(W, axis=0) # w_diff = w(t) - w(t-1)

# Collect data for the CBOE volatility index (measures expectation of stock market volatility for the S&P 500 index options)
start_date = "1999-06-30"
end_date = "2017-09-20"
vix = yf.download('^VIX', start=start_date, end=end_date)
print(vix.head())

# Set dependent variable as VIX at close
y = vix['Close']

# Construct X matrix
vix_dates = vix.index
statement_dates = x
X = np.zeros((len(y), W.shape[1]))
for i, vix_date in enumerate(vix_dates):
    for j, statement_date in enumerate(statement_dates):
        if int(date2num(vix_date)) == int(date2num(statement_date)):
            X[i,:] = W_diff[j,:]

# Estimate OLS
X_cons = sm.add_constant(X) # add column of ones/intercept
model = sm.OLS(y[1:], X_cons[1:,:]) # drop row with NaNs due to first differencing
results = model.fit()
print(results.summary())

