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
import datetime
import csv
from tfidf_basic_search import *
import gc



# ## Using LDA to rank documents
# LDA is optimized by coherence score u_mass



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




# Now we extract articles contain the most relevant topic

def selected_best_LDA(path, data, keyword, varname, num_topic):
        """Select the best lda model with extracted text """
        # convert data to dictionary format
        file_exists = os.path.isfile(path + 'result/lda_result.csv')
        f = open(path + 'result/lda_result.csv', 'a')
        writer_top = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer_top.writerow(['a'] + ['b'] + ['coherence'] + ['time'] +['topics'] )

        m = MetaData(data)
        metaDict = m.data_dict()

        #process text and extract text with keywords
        et = ExtractText(metaDict, keyword, varname)
        text1 = et.extract_w_keywords()


        # extract nouns, verbs and adjetives
        text = et.get_noun_verb2(text1)

        # optimized alpha and beta
        alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
        beta = [0.1, 0.3, 0.5, 0.7, 0.9]

        # alpha = [0.3, 0.9]
        # beta = [0.3, 0.9]

        mydict = lambda: defaultdict(mydict)
        cohere_dict = mydict()
        for a in alpha:
            for b in beta:
                lda = LDATopic(text, num_topic, a, b)
                model, coherence, scores = lda.topic_modeling()
                cohere_dict[coherence]['a'] = a
                cohere_dict[coherence]['b'] = b

                
        # sort result dictionary to identify the best a, b
        # select a,b with the largest coherence score 
        sort = sorted(cohere_dict.keys())[0] 
        a = cohere_dict[sort]['a']
        b = cohere_dict[sort]['b']
        
        # run LDA with the optimized values
        lda = LDATopic(text, num_topic, a, b)
        model, coherence, scores_best = lda.topic_modeling()
        #pprint(model.print_topics())

        #f = open(path + 'result/lda_result.csv', 'a')
        result_row = [[a, b, coherence, str(datetime.datetime.now()), model.print_topics()]]
        writer_top.writerows(result_row)

        f.close()
        gc.collect()
    
        # select merge ids with the LDA topic scores
        id_l = lda.get_ids_from_selected(text)
        scores_best['cord_uid'] = id_l


        return scores_best



# here we select the text with the most relevant topic according to the LDA result
def select_text_from_LDA_results(file, keyword, varname, scores_best, topic_num):
        # choose papers with the most relevant topic
        # convert data to dictionary format
        m = MetaData(file)
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
                print(v['cos_similarity'])
                selected[k]['cosine_similarity'] = v['cos_similarity']

        print ("There are {} abstracts selected". format(len(selected)))
        return selected

def extract_relevant_sentences(cor_dict, search_keywords, filter_title=None):
    """Extract sentences contain keyword in relevant articles. """
    #here user can also choose whether they would like to only select title contain covid keywords

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
                    sel_sentence[k]['cosine_similarity'] = v['cos_similarity']

        else:
            sel_sentence[k]['sentences'] = keyword_sentence
            sel_sentence[k]['sha'] = v['sha']
            sel_sentence[k]['title'] = v['title'] 
            sel_sentence[k]['cosine_similarity'] = v['cos_similarity']

            
    print('{} articles are relevant to the topic you choose'.format(len(sel_sentence)))
    return sel_sentence

def store_extract_sentences(sel_sentence, search_keywords):
    path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/search_results/'
    df = pd.DataFrame.from_dict(sel_sentence, orient='index')
    df.to_csv(path + 'search_results_{}.csv'.format(search_keywords))
    sel_sentence_df = pd.read_csv(path + 'search_results_{}.csv'.format(search_keywords))
    return sel_sentence_df


# ## Question 1: Is wearing mask an effective way to control pandemic?

# now we run topic search on teh test collection
#here we select the best LDA model  # keywords, search text, number of topic


path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_scripts/'

sr = BasicSearch('wear mask', 'abstract')
test_collection = sr.load_data()
# search_query_weights, tfidf_weights_matrix = sr.tf_idf(sr.search_keys, df, 'abstract')
# similarity_list = sr.cos_similarity(search_query_weights, tfidf_weights_matrix)

# #here we obtain the top n most similar 
# c = sr.most_similar(df, similarity_list)
# c.to_csv(sr.path + 'test.csv')

scores_best_mask = selected_best_LDA(path, 'test.csv', 'mask', 'abstract', 15)

# # topic number 1， 10 are the most relevant to public wearing mask
# which topic do you think is most relevant to your search
cor_dict_mask = select_text_from_LDA_results('test.csv', 'mask', 'abstract', scores_best_mask, 1)
cor_dict_mask2 = select_text_from_LDA_results('test.csv', 'mask', 'abstract', scores_best_mask, 10)
cor_dict_mask.update(cor_dict_mask2)
cor_dict_df = pd.DataFrame.from_dict(cor_dict_mask, orient='index')
cor_dict_df.to_csv(sr.path + 'test_selected.csv')


def evaluation_k(result, k, test_collection):
    '''get precision and recall at k'''
    # sort dictionary
    sort_df = result.sort_values(by=['cos_similarity'], ascending=False)
    top_k  = sort_df.head(k)
    top_k['system_label'] = 1
    # merge search result with all
    all_label = top_k.merge(test_collection, how = 'inner')
    #assign search result as 1
    all_label['system_label']
    #random assign true label
    report = classification_report(top_k['human'], top_k['machine'], output_dict=True)

# m = MetaData('test.csv')
# metaDict = m.data_dict()

# # process text and extract text with keywords
# et = ExtractText(metaDict, 'mask', 'abstract')
# # extract text together with punctuation
# text1 = et.extract_w_keywords_punc()





# # # We observe topic No. 14 is most relevant to public wearing mask, we can select multiple topics
# cor_dict_mask = select_text_from_LDA_results('mask', 'abstract', scores_best_mask, 9)



# # topic number 1， 5 is most relevant to public wearing mask
# # which topic do you think is most relevant to your search
# cor_dict_mask = select_text_from_LDA_results('mask', 'abstract', scores_best_mask, 13)
# print ("There are {} abstracts selected". format(len(cor_dict_mask)))
# cor_dict_mask2 = select_text_from_LDA_results('mask', 'abstract', scores_best_mask, 1)
# print ("There are {} abstracts selected". format(len(cor_dict_mask2)))
# cor_dict_mask.update(cor_dict_mask2)
# len(cor_dict_mask)


# # In[98]:


# # extract relevant sentences  #search keywords can be a list
# sel_sentence_mask = extract_relevant_sentences(cor_dict_mask, ['mask'])
# sel_sentence_df_mask = store_extract_sentences(sel_sentence_mask, 'mask')


# In[99]:


# #read extracted article
# sel_sentence_df_mask.head(10)


# # In[93]:


# def matching_labels(file1, file2, topic_name):
#     path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/annotation/'
#     path2 = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/search_results/'
#     incu1 = pd.read_csv(path + file1)
#     incu2 = pd.read_csv(path2 + file2)

#     incu1.rename(columns={incu1.columns[0]: "textid" }, inplace = True)
#     incu2.rename(columns={incu2.columns[0]: "textid" }, inplace = True)
#     incu = incu1.merge(incu2, on ='textid', how='right')
#     incu.to_csv(path2 + '{}_annotation.csv'.format(topic_name))
#     print(incu.shape)
    
# matching_labels('wear_mask.csv', 'search_results_mask.csv', 'mask')


# # ### Annotation guidline for question 1
# # We extracted 33 papers that are supposed to discuss whether using masks is useful. We annotate  whether the key sentences suggest using mask can reduce the risk of infection.
# # 
# # #### Stance Annotation 
# # * ‘1’ sentences that support using a mask during a pandemic is useful 
# # * ‘2’  papers that assume masks as useful and examine the public’s willingness to comply the rules,
# # * ’0’ no obvious evidence that shows using mask is protective or the protection is very little
# # * '3' Not relevant to the above stance
# # 
# # #### relevance annotation
# # * '1' the result is relevent to the question  
# # * '0' the result is not relevant to the question

# # In[48]:


# #here we need to add the stats analysis 
# path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/annotation/'
# annotation_mask = pd.read_csv(path + 'wear_mask.csv')


# # In[49]:


# # view file
# annotation_mask.head(5)
# print('there are {} articles relevant to the topic'.format(annotation_mask.shape[0]))


# # * ‘1’  support using a mask during a pandemic is useful 
# # * ‘2’  assume masks as useful and examine the public’s willingness to comply the rules,
# # * ’0’ no obvious evidence that shows using mask is protective or the protection is very little
# # * '3' Not relevant to the above stance
# # 
# # result from annotator 1

# # In[289]:


# annotation_mask['stance'].value_counts()


# # result from annotator 2

# # In[290]:


# annotation_mask['stance.1'].value_counts()


# # In[59]:


# annotation_mask['stance'].value_counts()[2]


# # In[64]:


# print('there are {} papers support using a mask during a pandemic is useful, {} assume masks as useful and examine the public’s willingness to comply the rules,  {} papers show no obvious evidence that shows using mask is protective or the protection is very little'. format(str(annotation_mask['stance'].value_counts()[1]), str(annotation_mask['stance'].value_counts()[2]), annotation_mask['stance'].value_counts()[0]) )
          


# # ### inter-rater repliability 

# # In[291]:


# cohen_kappa_score(annotation_mask['stance'], annotation_mask['stance.1'])


# # In[87]:


# mask = annotation_mask['relevance'].value_counts()
# print('there are {} papers relevant to the topic, {} papers not relevant to the topic'. format(mask[1], mask[0]))


# # ### First author location

# # In[ ]:





# # ## Results
# # According to the key sentences in 33 abstract that discuss the topic of public using masks, only one paper suggests that there’s not enough evidence to show that mask is useful.
# # There are 14 papers that suggest their results show using surgical mask during a pandemic is effective in reducing infection
# # 14 paper consider public individuals using masks are necessary in reducing risks of being infect, and these paper look at whether the public are willing to comply to the rules. (X papers are from  Hong Kong, based on the region of the first author)
# # 5 papers are not relevant to the topic
# # 
# # Conclusion:
# # government in some regions advocate using masks as a standard approach to reduce risk of infection, papers in these regions focus on whether people comply to the rules. When some government advocate that there is little evidence show that mask is effective in controlling the pandemic, nearly half of the academic papers from our search result either consider wearing masks as a standard practice that the public show comply, nearly half of the papers found evidence to support that wearing masks is effective in controlling the pandemic.
# # 

# # ### Question 2: How long in incubation period? In some region (e.g. China), there’s rumour circulating that the incubation period is longer than 14 days

# # ### Annotation guideline for question 2:
# # 
# # #### stance annotation
# # Here we want to identify papers that report a result aligns with the incubation period reported by the governments
# # UK government advocate: 2-14 days, mean 5
# # * ‘1’  same as government advocate 
# # * ‘0’  different from what the government
# # *  Not relevant to the question 
# # 
# # #### relevance annotation
# # * '1' the result is relevent to the question  
# # * '2' the result is not relevant to the question

# # In[134]:


# scores_best_incu = selected_best_LDA(['incubation'], 'abstract', 30)


# # In[127]:


# scores_best_incu.shape


# # In[126]:


# # topic number 0 is most relevant to public wearing mask
# # which topic do you think is most relevant to your search
# # cor_dict_incu = select_text_from_LDA_results('incubation', 'abstract', scores_best_incu, 26)
# # print ("There are {} abstracts selected". format(len(cor_dict_incu)))
# cor_dict_incu2 = select_text_from_LDA_results('incubation', 'abstract', scores_best_incu, 9)
# print ("There are {} abstracts selected". format(len(cor_dict_incu2)))
# cor_dict_incu3 = select_text_from_LDA_results('incubation', 'abstract', scores_best_incu, 1)
# print ("There are {} abstracts selected". format(len(cor_dict_incu3)))
# cor_dict_incu.update(cor_dict_incu2)
# cor_dict_incu.update(cor_dict_incu3)
# len(cor_dict_incu)


# # In[113]:


# # extract relevant sentences  #search keywords can be a list
# sel_sentence_incu, sel_sentence_df_incu = extract_relevant_sentences(cor_dict_incu, ['day'], 'title')


# # In[103]:


# #read extracted article
# sel_sentence_df_incu.head(10)


# # ## Incubation period statistical analysis

# # In[84]:


# #here we need to add the stats analysis 
# path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/annotation/'
# annotation_incubation = pd.read_csv(path + 'incubation.csv')
# print('there are {} articles relevant to the topic'.format(annotation_incubation.shape[0]))


# # result from annotator 1

# # In[85]:


# incubation = annotation_incubation['stance'].value_counts()
# print('there are {} paper shows the incubation period is 2-14 days with mean 5 days, {} papers shows a different number'. format(incubation[0], incubation[1])
#      )


# # In[86]:


# incubation = annotation_incubation['relevance'].value_counts()
# print('there are {} papers relevant to the topic, {} papers not relevant to the topic'. format(incubation[1], incubation[0]))


# # ## Question 3: Are asymptomatic patients infectious?
# # 

# # ### Annotation guideline for question 3:
# # Here we want to identify whether asymtomatic cases contribute to the spread of the virus
# # 
# # #### stance annotation
# # * ‘1’  there is clear evidence show that asymtomatic cases contribute to the spread of the virus
# # * ‘0’  it is unlikely that asymtomatic cases contribute to the spread of the virus
# # * '3' Not relevant to the question
# # 
# # #### relevance annotation
# # * '1' the result is relevent to the question  
# # * '0' the result is not relevant to the question

# # In[62]:


# scores_best_asym = selected_best_LDA('asymptomatic', 'abstract')


# # In[63]:


# # topic number 19 is most relevant to public wearing mask
# # which topic do you think is most relevant to your search
# cor_dict_asym = select_text_from_LDA_results('asymptomatic', 'abstract', scores_best_asym, 19)
# print ("There are {} abstracts selected". format(len(cor_dict_asym)))


# # In[119]:


# # extract relevant sentences  #search keywords can be a list
# sel_sentence_asym, sel_sentence_df_asym = extract_relevant_sentences(cor_dict_asym, ['transmission'], 'title')


# # In[120]:


# sel_sentence_df_asym.tail(10)


# # ## Asymptomatic Result

# # In[78]:


# #here we need to add the stats analysis 
# annotation_asymptomatic = pd.read_csv(path + 'asymtomatic.csv')
# print('there are {} articles relevant to the topic'.format(annotation_asymptomatic.shape[0]))


# # In[79]:


# asymptomatic = annotation_asymptomatic['stance'].value_counts()
# print('{} papers show that there is clear evidence show that asymtomatic cases contribute to the spread of the virus, {} papers show that it is unlikely that asymtomatic cases contribute to the spread of the virus'.format(asymptomatic[1], asymptomatic[0]))


# # In[82]:


# asymptomatic = annotation_asymptomatic['relevance'].value_counts()
# print('there are {} papers relevant to the topic, {} papers not relevant to the topic'. format(asymptomatic[1], asymptomatic[0]))


# # ## Question 4: Will the virus disappear in the summer? 
# # 
# # ### Annotation guideline for question 4
# # * '1' the result is relevent to the question  
# # * '0' the result is not relevant to the question

# # In[255]:


# scores_best_sea = selected_best_LDA('seasonality', 'abstract')


# # In[268]:


# # topic number 19 is most relevant to publicr wearing mask
# # which topic do you think is most relevant to your search
# cor_dict_sea = select_text_from_LDA_results('season', 'abstract', scores_best_sea, 0)
# print ("There are {} abstracts selected". format(len(cor_dict_sea)))


# # In[269]:


# # extract relevant sentences  #search keywords can be a list
# sel_sentence_sea , sel_sentence_df_sea  = extract_relevant_sentences(cor_dict_sea, ['summer'])


# # In[172]:


# sel_sentence_df_sea.tail(10)


# # ## virus and temperature result

# # In[72]:


# annotation_seasonality= pd.read_csv(path + 'seasonality.csv')
# print('there are {} articles relevant to the topic'.format(annotation_seasonality.shape[0]))


# # In[76]:


# seasonality = annotation_seasonality['relevance'].value_counts()
# print('there are {} papers relevant to the topic, {} papers not relevant to the topic'. format(seasonality[1], seasonality[0]))


