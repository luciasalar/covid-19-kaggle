import pandas as pd 
from collections import defaultdict
import string
from gensim.models import CoherenceModel
import gensim
from pprint import pprint
import spacy,en_core_web_sm
from nltk.stem import PorterStemmer
import os
import json
from gensim.models import Word2Vec
import nltk
import re
import collections
from sklearn.metrics import cohen_kappa_score
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import string
from preprocess import *

# # Building test collection
# 
# Here we use keyword approach to select abstracts that are associated with a statement and construct a test collection. Keyword approach allow us to include a wider collection of abstracts


def extract_relevant_sentences2(cor_dict, search_keywords, filter_title=None):
    """Extract sentences contain keyword in relevant articles for system evaluation. """
    #here user can also choose whether they would like to only select title contain covid keywords
    #difference from the previous one is where we store the result

    mydict = lambda: defaultdict(mydict)
    sel_sentence = mydict()
    filter_w = ['covid19','ncov','2019-ncov','covid-19','sars-cov','wuhan']
    
    for k, v in cor_dict.items():
        keyword_sentence = []
        sentences = v['processed_text'].split('.')
        for sentence in sentences:
            # for each sentence, check if keyword exist
            # append sentences contain keyword to list
            keyword_sum = sum(1 for word in search_keywords if word in sentence)
            if keyword_sum > 0:
                keyword_sentence.append(sentence)         

        # store results
        if not keyword_sentence:
            pass
        
        elif filter_title is not None:
            for f in filter_w:
                title = v['title'].lower().translate(str.maketrans('', '', string.punctuation))
                abstract = v['processed_text'].lower().translate(str.maketrans('', '', string.punctuation))
                if (f in title) or (f in abstract):
                    sel_sentence[k]['sentences'] = keyword_sentence
                    sel_sentence[k]['sha'] = v['sha']
                    sel_sentence[k]['title'] = v['title'] 
        else:
            sel_sentence[k]['sentences'] = keyword_sentence
            sel_sentence[k]['sha'] = v['sha']
            sel_sentence[k]['title'] = v['title'] 
    print('{} articles contain keyword {}'.format(len(sel_sentence),  search_keywords))

    path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/test_collection/'
    df = pd.DataFrame.from_dict(sel_sentence, orient='index')
    df.to_csv(path + 'test_{}.csv'.format(search_keywords))
    sel_sentence_df = pd.read_csv(path + 'test_{}.csv'.format(search_keywords))
    return sel_sentence, sel_sentence_df

def test_collection(keyword_l, varname, search_keywords, filter_title1=None):
    """Key word approach to extract abstracts for evaluation."""
    #process text and extract text with keywords
    m = MetaData('metadata.csv')
    metaDict = m.data_dict()
    et = ExtractText(metaDict, 'keyword_l', varname)
    text1 = et.extract_w_keywords_punc_multiplew(keyword_l)

    # filter out titles do not contain keywords
    if filter_title1 is not None:
        sel_sentence, sel_sentence_df = extract_relevant_sentences2(text1, search_keywords, 'title')
    else:
        sel_sentence, sel_sentence_df = extract_relevant_sentences2(text1, search_keywords)
   

# matching labels with test collection
def matching_labels(file1, file2, topic_name):
    path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/annotation/'
    path2 = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/test_collection/'
    incu1 = pd.read_csv(path + file1)
    incu2 = pd.read_csv(path2 + file2)

    incu1.rename(columns={incu1.columns[0]: "textid" }, inplace = True)
    incu2.rename(columns={incu2.columns[0]: "textid" }, inplace = True)
    incu = incu1.merge(incu2, on ='textid', how='right')
    incu.to_csv(path2 + '{}_test.csv'.format(topic_name))
    print(incu.shape)
    return incu


# collect abstracts that mention summer
test_collection('season', 'abstract', ['summer'])
# collect abstracts that mention mask
test_collection('mask', 'abstract', ['mask'])
# we only want sentences contain 'day' in incubation period related abstracts
test_collection('incubation', 'abstract', ['day'], 'title')
# collect abstracts mention asymptomatic and contagious
test_collection(['asymptomatic', 'contagious'], 'abstract', ['asymptomatic'], 'title')


# path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
# combine_test = combine_test[['textid']]
# combine_test = combine_test.rename(columns={"textid": "cord_uid"})
# meta_data = pd.read_csv(path +'metadata.csv')
# test_collection = combine_test.merge(meta_data, on = 'cord_uid', how = 'inner')
# print('size of test collection', test_collection.shape)
# test_collection.to_csv(path + 'test_collection.csv')


#here we match the test collection with data we annotated
# asymtomatic = matching_labels('asymtomatic.csv','test_asymptomatic.csv', 'asymtomatic')
# incubation = matching_labels('incubation.csv','test_incubation.csv', 'incubation')
# mask = matching_labels('wear_mask.csv','test_mask.csv', 'mask')
# season = matching_labels('seasonality.csv','test_season.csv', 'season')


# ### combine the test collection as one dataset

# combine_test = asymtomatic.append(incubation)
# combine_test = combine_test.append(mask)
# combine_test = combine_test.append(season)
# combine_test.shape
