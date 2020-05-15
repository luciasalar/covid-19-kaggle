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


# Read metadata into dictionary format
class MetaData:
    def __init__(self, data):
        """Define varibles."""
        # path and data
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.meta_data = pd.read_csv(self.path + data)

    def data_dict(self):
        """Convert df to dictionary. """
        mydict = lambda: defaultdict(mydict)
        meta_data_dict = mydict()

        if ('cos_similarity' in self.meta_data.columns) & ('abstract' in self.meta_data.columns):
            for cord_uid, abstract, title, sha, cos in zip(self.meta_data['cord_uid'], self.meta_data['abstract'], self.meta_data['title'], self.meta_data['sha'], self.meta_data['cos_similarity']):
                meta_data_dict[cord_uid]['title'] = title
                meta_data_dict[cord_uid]['abstract'] = abstract
                meta_data_dict[cord_uid]['sha'] = sha
                meta_data_dict[cord_uid]['cos_similarity'] = cos
 
        elif ('abstract' in self.meta_data.columns): 
            for cord_uid, abstract, title, sha in zip(self.meta_data['cord_uid'], self.meta_data['abstract'], self.meta_data['title'], self.meta_data['sha']):
                meta_data_dict[cord_uid]['title'] = title
                meta_data_dict[cord_uid]['abstract'] = abstract
                meta_data_dict[cord_uid]['sha'] = sha
        else:
            for cord_uid, abstract, title, sha in zip(self.meta_data['cord_uid'], self.meta_data['processed_text'], self.meta_data['title'], self.meta_data['sha']):
                meta_data_dict[cord_uid]['title'] = title
                meta_data_dict[cord_uid]['processed_text'] = abstract
                meta_data_dict[cord_uid]['sha'] = sha


        return meta_data_dict



# Extract documents contain keywords, preprocessing 

class ExtractText:
    """Extract text according to keywords or phrases"""

    def __init__(self, metaDict, keyword, variable):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.metadata = metaDict
        self.keyword = keyword
        self.variable = variable


    def simple_preprocess(self):
        """Simple text process: lower case, remove punc. """
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        for k, v in self.metadata.items():
            sent = v[self.variable]
            sent = str(sent).lower().translate(str.maketrans('', '', string.punctuation))
            cleaned[k]['processed_text'] = sent
            cleaned[k]['sha'] = v['sha']
            cleaned[k]['title'] = v['title']

        return cleaned

    def very_simple_preprocess(self):
        """Simple text process: lower case only. """
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        # check if there's cosine similarity, we need to include it
        for k, v in self.metadata.items():
            sent = v[self.variable]
            sent = str(sent)
            if 'cos_similarity' in self.metadata[list(self.metadata.keys())[0]].keys():   
                cleaned[k]['processed_text'] = sent
                cleaned[k]['sha'] = v['sha']
                cleaned[k]['title'] = v['title']
                cleaned[k]['cos_similarity'] = v['cos_similarity']

            else:
                cleaned[k]['processed_text'] = sent
                cleaned[k]['sha'] = v['sha']
                cleaned[k]['title'] = v['title']

        return cleaned

   

    def very_simple_preprocess_stem(self):
        """Simple text process: lower case only. """
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        # check if there's cosine similarity, we need to include it
        if 'cos_similarity' in self.metadata[list(self.metadata.keys())[0]].keys():
            for k, v in self.metadata.items():
                sent = v[self.variable]
                sent = str(sent)
                cleaned[k]['processed_text'] = sent
                cleaned[k]['sha'] = v['sha']
                cleaned[k]['title'] = v['title']
                cleaned[k]['cos_similarity'] = v['cos_similarity']

        else:
            for k, v in self.metadata.items():
                sent = v[self.variable]
                cleaned[k]['processed_text'] = ps.stem(str(sent).split())
                cleaned[k]['sha'] = v['sha']
                cleaned[k]['title'] = v['title']

        return cleaned

    def extract_w_keywords(self):
        """Select content with keywords."""
        ps = PorterStemmer()
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.simple_preprocess()
        
        for k, v in textdict.items():
            for keyw in self.keyword:
                if ps.stem(str(keyw)) in ps.stem(str(v['processed_text'].split())):
                    #print(ps.stem(str(self.keyword)))
                    selected[k]['processed_text'] = v['processed_text']
                    selected[k]['sha'] = v['sha']
                    selected[k]['title'] = v['title']
        return selected

    def extract_w_keywords_punc(self):
        """Select content with keywords, with punctuations in text"""
        ps = PorterStemmer()
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.very_simple_preprocess()
        # check if there's cosine similarity, we need to include it
        if 'cos_similarity' in self.metadata[list(self.metadata.keys())[0]].keys():
            for k, v in textdict.items():
                if ps.stem(str(self.keyword)) in ps.stem(str(v['processed_text'].split())):
                   # print('yes')    
                    selected[k]['processed_text'] = v['processed_text']
                    selected[k]['sha'] = v['sha']
                    selected[k]['title'] = v['title']
                    selected[k]['cos_similarity'] = v['cos_similarity']
            #keywords are stemmed before matching
            else:
                if ps.stem(str(self.keyword)) in ps.stem(str(v['processed_text'].split())):
                    selected[k]['processed_text'] = v['processed_text']
                    selected[k]['sha'] = v['sha']
                    selected[k]['title'] = v['title']
                    
        return selected
    
    def extract_w_keywords_punc_multiplew(self, keyword_l):
        """Select content with multiple keywords, with punctuations in text"""
        ps = PorterStemmer()
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.very_simple_preprocess()

 
        for k, v in textdict.items():
            for keyw in keyword_l:
                #keywords are stemmed before matching
                if ps.stem(str(keyw)) in ps.stem(str(v['processed_text'].split())):
                    selected[k]['processed_text'] = v['processed_text']
                    selected[k]['sha'] = v['sha']
                    selected[k]['title'] = v['title']
        return selected


    def get_noun_verb(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for LDA"""
        ps = PorterStemmer()
      
        #find nound trunks
        nlp = en_core_web_sm.load()
        all_extracted = {}
        for k, v in text.items():
            #v = v.replace('incubation period', 'incubation_period')
            doc = nlp(v)
            nouns = ' '.join(str(v) for v in doc if v.pos_ is 'NOUN').split()
            verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
            adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
            all_w = nouns + verbs + adj
            all_extracted[k] = all_w
      
        return all_extracted

    def get_noun_verb2(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for LDA"""
        ps = PorterStemmer()
      
        #find nound trunks
        nlp = en_core_web_sm.load()
        all_extracted = {}
        for k, v in text.items():
            #v = v.replace('incubation period', 'incubation_period')
            doc = nlp(v['processed_text'])
            nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split()
            verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
            adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
            all_w = nouns + verbs + adj
            all_extracted[k] = all_w
      
        return all_extracted

    def preprocess_cluster_sentence(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for LDA"""
        ps = PorterStemmer()
      
        #find nound trunks
        nlp = en_core_web_sm.load()
        all_extracted = {}
        for k, v in text.items():
            #v = v.replace('incubation period', 'incubation_period')
            doc = nlp(v['processed_text'])
            nouns = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'NOUN').split()
            verbs = ' '.join(ps.stem(str(v)) for v in doc if v.pos_ is 'VERB').split()
            adj = ' '.join(str(v) for v in doc if v.pos_ is 'ADJ').split()
            adv = ' '.join(str(v) for v in doc if v.pos_ is 'ADV').split()
            part = ' '.join(str(v) for v in doc if v.pos_ is 'PART').split()
            all_w = nouns + verbs + adj + adv + part
            all_extracted[k] = all_w
      
        return all_extracted

    def tokenization(self, text):
        """get noun trunks for the lda model,
        change noun and verb part to decide what
        you want to use as input for the next step"""
        nlp = spacy.load("en_core_web_sm")

        all_extracted = {}
        for k, v in text.items():
            doc = nlp(v)
            all_extracted[k] = [w.text for w in doc]
      
        return all_extracted
