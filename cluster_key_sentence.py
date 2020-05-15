from tfidf_basic_search import *
from topic_search import *
from test_collection import *




class ClusterSentence:
    """Here we cluster the key sentence from the target documents"""

    def __init__(self, keywords, outputfile, num_topics, search_results, eva_file, temp_eva_file, annotation_file, selected_topics):
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_scripts/'
        self.path2 = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.keywords = keywords #'wear mask'
        self.outputfile = outputfile #'wear_mask'
        self.num_topics = num_topics
        self.search_results = search_results #'search_results/basic_search_wear_mask.csv'
        self.eva_file = eva_file # 'mask_stem'
        self.temp_eva_file = temp_eva_file #'mask' name for temp evaluation file
        self.annotation_file = annotation_file 
        self.selected_topics = selected_topics # [8, 1, 3]
        

    
    def extract_documents(self, var):
        """Extract target documents and store result."""

        basic_search_by_word(self.keywords, self.outputfile)
        scores_best_mask = selected_best_LDA(self.path, self.search_results, 'abstract', self.num_topics)
        eva = Evaluation('/search_results/test_selected.csv', self.eva_file) #evaluation, test_selected.csv is result of the retrieval system #name of evaluation file
        plot_result, result = eva.evaluation(scores_best_mask, self.search_results, self.temp_eva_file, 'abstract', self.annotation_file, 'other_ignore', self.selected_topics)

        return result

    def extract_key_sentences(self, search_keywords):
        """Extract sentences with keywords"""
        m = MetaData('search_results/{}_temp_eva.csv'.format(self.temp_eva_file))
        metaDict = m.data_dict()
        et = ExtractText(metaDict, 'anything', 'processed_text')
        #text1 = et.very_simple_preprocess()
        sel_sentence, sel_sentence_df = extract_relevant_sentences2(metaDict, search_keywords)
        sel_sentence_df.rename(columns={sel_sentence_df.columns[0]: "cord_uid"}, inplace=True)
        sel_sentence_df.rename(columns={sel_sentence_df.columns[1]: "processed_text"}, inplace=True)
        sel_sentence_df.to_csv(self.path2 + 'temp.csv')
        return sel_sentence, sel_sentence_df

    def pre_process(self):
        m = MetaData('temp.csv')
        metaDict = m.data_dict()

        t = pd.read_csv(m.path + 'temp.csv')

        c = t['processed_text'].str.strip().str.lower().str.replace('[', '')
        c = c.str.strip().str.lower().str.replace(']', '')
        c = c.str.strip().str.lower().str.replace(',', '')
        t['processed_text'] = c.str.strip().str.lower().str.replace("'", '')
        t.to_csv(m.path + 'temp.csv')

    def cluster_key_sentences(self, num_topics):
        """Cluster key sentences """
        self.pre_process()
        #scores_best_mask = selected_best_LDA('temp.csv', 'varname', 5)
        scores_best_mask = selected_best_LDA_allw(self.path, 'temp.csv', 'processed_text', num_topics)
        return scores_best_mask

        


cl = ClusterSentence('wear mask', 'wear_mask', 20, 'search_results/basic_search_wear_mask.csv', 'mask_stem', 'mask', 'test_collection/labels/test_collection_mask_relabel.csv', [8, 1, 3])


#cluster sentences with keywords
sel_sentence, sel_sentence_df = cl.extract_key_sentences(['mask', 'wear'])
scores_best_mask = cl.cluster_key_sentences(10)

#further remove docs according to topics
path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_scripts/'
eva = Evaluation('/search_results/test_selected.csv', 'mask_sentence_clu') #evaluation, test_selected.csv is result of the retrieval system
plot_result, result = eva.evaluation_sent_cluster(scores_best_mask, 'temp.csv', 'mask', 'abstract', 'test_collection/labels/test_collection_mask_relabel.csv', 'topic', [1, 4])


# cor_dict_mask = select_text_from_LDA_results('temp.csv', 'anything', 'processed_text', scores_best_mask)

# result = pd.DataFrame.from_dict(cor_dict_mask, orient='index')
# result['cord_uid'] = result.index
# path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/search_results/'
# result.to_csv(path + 'test_selected.csv')

# sr = BasicSearch('wear mask', 'abstract')
# test_collection = sr.load_data(filename)

# #select target topic and put them at the back, here we want to put the dominant topics at the back
# result = result.sort_values(by=['Perc_Contribution'], ascending=[True])
# print(result.columns)

















