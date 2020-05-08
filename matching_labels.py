import pandas as pd 


"""Here we match the test collection with the label we have """

class MatchFiles:
    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/annotation/'
        self.path_tc = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/'
        self.path_tc_label = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/test_collection/labels/'

    def extract_relevance_labels(self, filename):
        """Extract relevant labels from annotated files. """
        file = pd.read_csv(self.path + filename, encoding="ISO-8859-1", engine='python')
        if 'textid' not in file.columns:
            file.rename(columns={file.columns[0]: "cord_uid"}, inplace=True)
        else:
            file = file[['textid', 'relevance']]
            file.rename(columns={file.columns[0]: "cord_uid"}, inplace=True)
        return file

    def match_testcollection(self, filename):
        """Here we combine the annotated files with test collection to generate annotation labels"""
        mask = self.extract_relevance_labels(filename)
        tc = pd.read_csv(self.path_tc + 'test_collection.csv')
        tc = tc[['cord_uid', 'abstract']]
        labels = mask.merge(tc, on='cord_uid', how='outer')
        labels['relevance'] = labels['relevance'].fillna(0)
        labels.to_csv(self.path_tc_label + 'test_collection_{}'.format(filename))
        return labels



m = MatchFiles()
all_files = m.match_testcollection('wear_mask.csv')



# def matching_labels(file1, file2, topic_name):
#     path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/annotation/'
#     path2 = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/cov19_2/search_results/'
#     incu1 = pd.read_csv(path + file1)
#     incu2 = pd.read_csv(path2 + file2)

#     incu1.rename(columns={incu1.columns[0]: "textid" }, inplace = True)
#     incu2.rename(columns={incu2.columns[0]: "textid" }, inplace = True)
#     incu = incu1.merge(incu2, on ='textid', how='right')
#     incu.to_csv(path2 + '{}_adjusted.csv'.format(topic_name))
#     print(incu.shape)





# #here we match test collection with data we annotated
# asymtomatic = matching_labels('asymtomatic.csv','test_asymptomatic.csv', 'asymtomatic')
# incubation = matching_labels('incubation.csv','test_incubation.csv', 'incubation')
# mask = matching_labels('wear_mask.csv','test_mask.csv', 'mask')
# season = matching_labels('seasonality.csv','test_season.csv', 'season')


# ### combine the test collection as one dataset

# combine_test = asymtomatic.append(incubation)
# combine_test = combine_test.append(mask)
# combine_test = combine_test.append(season)
# combine_test.shape

# # In[101]:


# # incubation
# matching_labels('incubation.csv', 'tfidf_incubation.csv', 'incubation')


# # In[104]:


# matching_labels('asymptomatic.csv', 'tfidf_asymptomatic.csv', 'asymptomatic')
