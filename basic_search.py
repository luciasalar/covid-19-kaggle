import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import string


class BasicSearch:
    """Basic search using tfidf and cosine similarity """
    def __init__(self, search_keys, varname):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.search_keys = search_keys
        self.variable = varname

    def load_data(self):
        """Load meta data."""
        dataframe = pd.read_csv(self.path + 'metadata.csv')
        return dataframe

    def tf_idf(self, search_keys, dataframe, varname):
        """Compute search query weights and tfidf weights."""
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_weights_matrix = tfidf_vectorizer.fit_transform(dataframe[varname].values.astype('U'))
        search_query_weights = tfidf_vectorizer.transform([self.search_keys])

        return search_query_weights, tfidf_weights_matrix

    def cos_similarity(self, search_query_weights, tfidf_weights_matrix):
        """Compute cosine similarity between weights """
        cosine_distance = cosine_similarity(search_query_weights, tfidf_weights_matrix)
        similarity_list = cosine_distance[0]

        return similarity_list


    def most_similar(self, similarity_list, min_talks=1):
        """Return entries with cosine similarity > 0 """
        most_similar = []
        for idx, num in enumerate(similarity_list):
            if num > 0:
                most_similar.append(idx)

        return most_similar

    def get_search_result(self):
        """Get search resutls. """
        df = self.load_data()
        search_query_weights, tfidf_weights_matrix = self.tf_idf(self.search_keys, df, 'abstract')
        similarity_list = s.cos_similarity(search_query_weights, tfidf_weights_matrix)

        c = s.most_similar(similarity_list, min_talks=1)
        df['index'] = df.index

        result_id = pd.DataFrame(c)
        result_id.rename(columns={result_id.columns[0]: "index" }, inplace = True)

        result = result_id.merge(df, on='index', how='inner')
        #add filter title
        result.to_csv(self.path + 'tfidf_search.csv')
        return result

    def convert_result_to_dict(self):
        """Convert result to dictionary. """
        result = self.get_search_result()
        mydict = lambda: defaultdict(mydict)
        result_data_dict = mydict()

        for cord_uid, abstract, title, sha in zip(result['cord_uid'], result['abstract'], result['title'], result['sha']):
            result_data_dict[cord_uid]['title'] = title
            result_data_dict[cord_uid]['abstract'] = abstract
            result_data_dict[cord_uid]['sha'] = sha

        return result_data_dict


    def simple_preprocess(self, result):
        """Simple text process: lower case, remove punc. """
        mydict = lambda: defaultdict(mydict)
        cleaned = mydict()
        for k, v in result.items():
            sent = v[self.variable]
            sent = str(sent).lower().translate(str.maketrans('', '', string.punctuation))
            cleaned[k]['processed_text'] = sent
            cleaned[k]['sha'] = v['sha']
            cleaned[k]['title'] = v['title']
            cleaned[k]['abstract'] = v[self.variable]

        return cleaned

    def extract_relevant_sentences(self, search_keywords, filter_title=None):
        """Extract sentences contain keyword in relevant articles. """
        #here user can also choose whether they would like to only select title contain covid keywords
        result_data_dict = self.convert_result_to_dict()
        processed_result = self.simple_preprocess(result_data_dict)

        mydict = lambda: defaultdict(mydict)
        sel_sentence = mydict()
        filter_w = ['cov', 'covid19', 'ncov', 'coronavirus', '2019-ncov', 'covid-19', 'mers-cov','sars-cov']
        
        for k, v in processed_result.items():
            keyword_sentence = []
            sentences = v['abstract'].split('.')
            for sentence in sentences:
                # for each sentence, check if keyword exist
                # append sentences contain keyword to list
                keyword_sum = sum(1 for word in search_keywords if word in sentence.lower())
                if keyword_sum > 0:
                    keyword_sentence.append(sentence)

            # store results
            if not keyword_sentence:
                pass
            elif filter_title is not None:
                for f in filter_w:
                    title = v['title'].lower().translate(str.maketrans('', '', string.punctuation))
                    abstract = v['abstract'].lower().translate(str.maketrans('', '', string.punctuation))
                    if (f in title) or (f in abstract):
                        sel_sentence[k]['sentences'] = keyword_sentence
                        sel_sentence[k]['sha'] = v['sha']
                        sel_sentence[k]['title'] = v['title']
            else:
                sel_sentence[k]['sentences'] = keyword_sentence
                sel_sentence[k]['sha'] = v['sha']
                sel_sentence[k]['title'] = v['title']

        print('{} articles are relevant to the topic you choose'.format(len(sel_sentence)))

        path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/search_results/'
        df = pd.DataFrame.from_dict(sel_sentence, orient='index')
        df.to_csv(path + 'tfidf_results_{}.csv'.format(search_keywords))
        sel_sentence_df = pd.read_csv(path + 'tfidf_results_{}.csv'.format(search_keywords))
        return sel_sentence, sel_sentence_df





s = BasicSearch('wear mask', 'abstract') #enter query
result = s.extract_relevant_sentences(['mask'])


# df = s.load_data()

# search_query_weights, tfidf_weights_matrix = s.tf_idf('wear mask', df, 'abstract')
# similarity_list = s.cos_similarity(search_query_weights, tfidf_weights_matrix)

# c = s.most_similar(similarity_list, min_talks=1)
# df['index'] = df.index

# result_id = pd.DataFrame(c) 
# result_id.rename(columns={result_id.columns[0]: "index" }, inplace = True)

# result = result_id.merge(df, on = 'index', how='inner')
# #add filter title
# result.to_csv(path + 'tfidf_search.csv')






















