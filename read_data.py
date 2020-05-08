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

class MetaData:
    def __init__(self):
        """Define varibles."""
        # path and data
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.meta_data = pd.read_csv(self.path + 'metadata.csv')

    def data_dict(self):
        """Convert df to dictionary. """
        mydict = lambda: defaultdict(mydict)
        meta_data_dict = mydict()

        for cord_uid, abstract, title, sha in zip(self.meta_data['cord_uid'], self.meta_data['abstract'], self.meta_data['title'], self.meta_data['sha']):
            meta_data_dict[cord_uid]['title'] = title
            meta_data_dict[cord_uid]['abstract'] = abstract
            meta_data_dict[cord_uid]['sha'] = sha

        return meta_data_dict



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
        for k, v in self.metadata.items():
            sent = v[self.variable]
            sent = str(sent).lower()
            cleaned[k]['processed_text'] = sent
            cleaned[k]['sha'] = v['sha']
            cleaned[k]['title'] = v['title']

        return cleaned
     

    def extract_w_keywords(self):
        """Select content with keywords."""
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.simple_preprocess()
        for k, v in textdict.items():
            if self.keyword in v['processed_text'].split():
                #print(v['sha'])
                selected[k]['processed_text'] = v['processed_text']
                selected[k]['sha'] = v['sha']
                selected[k]['title'] = v['title']
        return selected

    def extract_w_keywords_punc(self):
        """Select content with keywords, with punctuations in text"""
        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        textdict = self.very_simple_preprocess()
        for k, v in textdict.items():
            if self.keyword in v['processed_text'].split():
                    #print(v['sha'])
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



class LDATopic:
    def __init__(self, processed_text, topic_num, alpha, eta):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.text = processed_text
        self.topic_num = topic_num
        self.alpha = alpha
        self.eta = eta

    def get_lda_score_eval(self, dictionary, bow_corpus):
        """LDA model and coherence score."""

        lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=self.topic_num, id2word=dictionary, passes=10,  update_every=1, random_state = 300, alpha=self.alpha, eta=self.eta)
        #pprint(lda_model.print_topics())

        # get coherence score
        cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print('coherence score is {}'.format(coherence))

        return lda_model, coherence

    def get_score_dict(self, bow_corpus, lda_model_object):
        """
        get lda score for each document
        """
        all_lda_score = {}
        for i in range(len(bow_corpus)):
            lda_score ={}
            for index, score in sorted(lda_model_object[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                lda_score[index] = score
                od = collections.OrderedDict(sorted(lda_score.items()))
            all_lda_score[i] = od
        return all_lda_score


    def topic_modeling(self):
        """Get LDA topic modeling."""
        # generate dictionary
        dictionary = gensim.corpora.Dictionary(self.text.values())
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.text.values()]
        # modeling
        model, coherence = self.get_lda_score_eval(dictionary, bow_corpus)

        lda_score_all = self.get_score_dict(bow_corpus, model)

        all_lda_score_df = pd.DataFrame.from_dict(lda_score_all)
        all_lda_score_dfT = all_lda_score_df.T
        all_lda_score_dfT = all_lda_score_dfT.fillna(0)

        return model, coherence, all_lda_score_dfT

    def get_ids_from_selected(self, text):
        """Get unique id from text """
        id_l = []
        for k, v in text.items():
            id_l.append(k)
            
        return id_l



class MatchArticleBody:
    def __init__(self, path, selected_id):
        """Define varibles."""
        self.path = path
        self.selected_id = selected_id


    def read_folder(self):
        """
        Creates a nested dictionary that represents the folder structure of rootdir
        """
        rootdir = self.path.rstrip(os.sep)

        article_dict = {}
        for path, dirs, files in os.walk(rootdir):
            for f in files:
                file_id = f.split('.')[0]
                #print(file_id)
                try:
                # load json file according to id
                    with open(self.path + f) as f:
                        data = json.load(f)
                except:
                    pass
                article_dict[file_id] = data

        return article_dict


    def extract_bodytext(self):
        """Unpack nested dictionary and extract body of the article"""
        body = {}
        article_dict = self.read_folder()
        for k, v in article_dict.items():
            strings = ''
            prevString = ''
            for entry in v['body_text']:
                strings = strings + prevString
                prevString = entry['text']

            body[k] = strings
        return body


    def get_title_by_bodykv(self, article_dict, keyword):
        """Search keyword in article body and return title"""

        article_dict = self.read_folder()
        selected_id = self.extract_id_list()

        result = {}
        for k, v in article_dict.items():
            for entry in v['body_text']:
                if (keyword in entry['text'].split()) and (k in selected_id):
                    result[k] = v['metadata']['title']

        return result


    def extract_id_list(self):
        """Extract ids from the selected text. """
        selected_id = []
        for k, v in self.selected_id.items():
            selected_id.append(str(v['sha']).split(';')[0])
            try:
                selected_id.append(str(v['sha']).split(';')[1])
                selected_id.append(str(v['sha']).split(';')[2])
                selected_id.append(str(v['sha']).split(';')[3])
            except:
                pass

        return selected_id


    def select_text_w_id(self):
        body_text = self.extract_bodytext()
        selected_id = self.extract_id_list()
        selected_text = {}
        for k, v in body_text.items():
            if k in selected_id:
                selected_text[k] = v
        return selected_text



class word2Vec:
    """Here we use word2vec to view the word similarities."""
    def __init__(self, clean_text, keyword):
        self.clean_text = clean_text
        self.keyword = keyword

    def train_model(self):
        """train model and return word embeddings"""
        model = Word2Vec(self.clean_text.values(), min_count=1)
        word_vectors = model.wv

        return word_vectors

    def get_top_similar(self):
        """get the top similar words"""
        #clean_text = self.clean_text.values()
        word_vectors = self.train_model()
        result = word_vectors.most_similar(positive=[self.keyword], topn=20)
        print(result)

        return result



# m = MetaData()
# metaDict = m.data_dict()

# et = ExtractText(metaDict, 'mask', 'abstract') #lowercase
# #process text
# #text = et.extract_w_keywords()
# text = et.extract_w_keywords_punc()


#match articles from the db
#ma = MatchArticleBody('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/biorxiv_medrxiv/biorxiv_medrxiv/', text)
#articles = ma.read_folder()
#title = ma.get_title_by_bodykv(text, 'masks')


#body_text = ma.extract_bodytext()
#id_l = ma.extract_id_list()
# body_text = ma.select_text_w_id()
# body_text = et.get_noun_verb(body_text)
# #body_text = et.tokenization(body_text)
# # body_text = et.get_bigram(body_text) 

# wv = word2Vec(body_text, 'incubation')
# s = wv.get_top_similar()

# m = MetaData()
# metaDict = m.data_dict()

# #process text and extract text with keywords
# et = ExtractText(metaDict, 'mask', 'abstract')
# text = et.extract_w_keywords()
# lda = LDATopic(text, 20, 0.9, 0.9)


# Now we extract articles contain the most relevant topic

def selected_best_LDA(keyword, varname):
        """Select the best lda model with extracted text """
        # convert data to dictionary format
        m = MetaData()
        metaDict = m.data_dict()

        #process text and extract text with keywords
        et = ExtractText(metaDict, keyword, varname)
        text1 = et.extract_w_keywords()


        # extract nouns, verbs and adjetives
        text = et.get_noun_verb2(text1)

        # optimized alpha and beta
        alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
        beta = [0.1, 0.3, 0.5, 0.7, 0.9]

        mydict = lambda: defaultdict(mydict)
        cohere_dict = mydict()
        for a in alpha:
            for b in beta:
                lda = LDATopic(text, 20, a, b)
                model, coherence, scores = lda.topic_modeling()
                cohere_dict[coherence]['a'] = a
                cohere_dict[coherence]['b'] = b
    
        # sort result dictionary to identify the best a, b
        # select a,b with the largest coherence score 
        sort = sorted(cohere_dict.keys())[0] 
        a = cohere_dict[sort]['a']
        b = cohere_dict[sort]['b']
        
        # run LDA with the optimized values
        lda = LDATopic(text, 20, a, b)
        model, coherence, scores_best = lda.topic_modeling()
        pprint(model.print_topics())

        # select merge ids with the LDA topic scores
        id_l = lda.get_ids_from_selected(text)
        scores_best['cord_uid'] = id_l

        return scores_best



def select_text_from_LDA_results(keyword, varname, scores_best, topic_num):
        # choose papers with the most relevant topic
        # convert data to dictionary format
        m = MetaData()
        metaDict = m.data_dict()

        # process text and extract text with keywords
        et = ExtractText(metaDict, keyword, varname)
        # extract text together with punctuation
        text1 = et.extract_w_keywords_punc()
        # need to decide which topic to choose after training
        sel = scores_best[scores_best[topic_num] > 0] 
        

        mydict = lambda: defaultdict(mydict)
        selected = mydict()
        for k, v in text1.items():
            if k in sel.cord_uid.tolist():
                selected[k]['title'] = v['title']
                selected[k]['processed_text'] = v['processed_text']
                selected[k]['sha'] = v['sha']

        return selected

def extract_relevant_sentences(cor_dict, search_keywords):
    """Extract sentences contain keyword in relevant articles. """

    mydict = lambda: defaultdict(mydict)
    sel_sentence = mydict()
    
    for k, v in cor_dict.items():
        keyword_sentence = []
        sentences = v['processed_text'].split('.')
        for sentence in sentences:
            # for each sentence, check if keyword exist
            # append sentences contain keyword to list
            keyword_sum = sum(1 for word in search_keywords if word in sentence)
            if (keyword_sum > 0) and ('COVID-19' in v['title']):
                print('yes')
                keyword_sentence.append(sentence)
            

        # store results
        sel_sentence[k]['sentences'] = keyword_sentence
        sel_sentence[k]['sha'] = v['sha']
        sel_sentence[k]['title'] = v['title']
    print('{} articles are relevant to the topic you choose'.format(len(sel_sentence)))

    path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
    df = pd.DataFrame.from_dict(sel_sentence, orient='index')
    df.to_csv(path + 'search_results_{}.csv'.format(search_keywords))
    return sel_sentence



scores_best = selected_best_LDA('infectious', 'abstract')
# topic number 1 is most relevant to public wearing mask
# which topic do you think is most relevant to your search
#cor_dict = select_text_from_LDA_results('asymptomatic', 'abstract', scores_best, 19)
# # extract relevant sentences  #search keywords can be a list
#sel_sentence = extract_relevant_sentences(cor_dict, ['asymptomatic'])








def run_lda_other_text():
    """Run lda model."""
    # convert data to dictionary
    m = MetaData()
    metaDict = m.data_dict()

    #process text and extract text with keywords
    et = ExtractText(metaDict, 'mask', 'abstract')
    text = et.extract_w_keywords()

    # extract nouns
    text = et.get_noun_verb2(text)
    #run topic model
    lda = LDATopic(text, 20, 0.9, 0.9)
    m, c, scores = lda.topic_modeling()

    return m, c, scores

#m, c, s = run_lda_other_text()


def run_lda_body_text():
    """Run lda model."""
    # convert data to dictionary
    m = MetaData()
    metaDict = m.data_dict()

    #process text and extract text with keywords
    et = ExtractText(metaDict, 'mask', 'abstract')
    text = et.extract_w_keywords()

    #match articles from the db
    ma = MatchArticleBody('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/biorxiv_medrxiv/biorxiv_medrxiv/', text)
    body_text = ma.select_text_w_id()

    # extract nouns 
    text = et.get_noun_verb2(text)
    #run topic model
    lda = LDATopic(text, 20, 0.9, 0.9)
    m, c, s = lda.topic_modeling()

    return m, c

#m, c = run_lda_body_text()


def run_w2v_body_text(search_kv, search_var, cosine_kv):
    """
    Extract text according to keywords in abstract, then select keywords in 
    article body and identify words with clostest distance.
    """
    m = MetaData()
    metaDict = m.data_dict()

    et = ExtractText(metaDict, search_kv, search_var) #lowercase
    #process text
    text = et.extract_w_keywords()

    #match articles from the db
    ma = MatchArticleBody('/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/biorxiv_medrxiv/biorxiv_medrxiv/', text)
    body_text = ma.select_text_w_id()
    # extract noun, verb and adj only, verbs are stemmed
    body_text = et.get_noun_verb(body_text)
    # word2vec embeddings and cosine similarity
    wv = word2Vec(body_text, cosine_kv)
    s = wv.get_top_similar()

    return s

#run_w2v_body_text('china', 'abstract', 'incubation')



def run_w2v(search_kv, search_var, cosine_kv):
    """
    Extract text according to keywords in abstract, then select keywords in 
    text and identify words with clostest distance.
    """
    m = MetaData()
    metaDict = m.data_dict()

    et = ExtractText(metaDict, search_kv, search_var) #lowercase
    #process text
    text = et.extract_w_keywords()

    # extract noun, verb and adj only, verbs are stemmed
    text = et.get_noun_verb2(text)
    # word2vec embeddings and cosine similarity
    wv = word2Vec(text, cosine_kv)
    s = wv.get_top_similar()

    return s


#s = run_w2v('mask', 'abstract', 'transmission')





