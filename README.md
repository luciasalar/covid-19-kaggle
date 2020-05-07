# Fighting COVID-19 Infodemics 
## Project Design

Knowledge gap between public and specialists and uncertainties are an important factors that drive pandemic anxiety, in this task, we will examine papers that discuss some of the controversial topics that contribute to rumours and anxiety

Here I develop a search system to extract sentences from abstracts that are relevant to a question. The questions are associated with rumour and uncertain information circulating in the public. We can try different questions in here, and an important part is to evaluate the search system with human annotation baseline if we want to push forward this work as a paper. 

### Step 1:
The search system first extract abstract contains a keyword (e.g. ‘mask’), then we use LDA to group the abstract topics. We identify a topic that is  most relevant to the question and we extract abstracts that contain the target topic. The system sentences that contain the keyword from the relevant abstracts. The standard apporach of a search system is to used TFIDF to rank documents, here we use LDA topic modeling on nouns, verbs and adjectives of the abstract. Users can decide the relevant information when they know what are the most frequent keywords in each topic. For some queries, users want to identify articles for covid-19 only. Therefore, users are opt to add title filer for different queries in our system.

The benefit of this approach is that when we want to know the relevant content for a question, we don't know what are the keywords in the article are more relevant to the question we ask, because the users are usually not farmiliar with academic papers. In our system, the topic keywords serve as prime for the query in the next step for extracting sentences in the abstract.

### Step 2:
We manually annotate the key sentences to identify the stance of the result sentences and whether these sentences are relevant to the question asked.


Keywords:
Incubation period, asymptomatic, mask, death rate, paracetamone


### Annotation
To understand the answer to the relevant question, we need to annotate the stance of the results, such as, does the abstract for / against the statement. 

To evaluate the search system, we need to annotate the relevance of the retrieved result. Please refer to each section for annotation guildline

Retrieved results and annotations are in this document 
https://docs.google.com/spreadsheets/d/1-eWEqji7mLXNF0Z9KH8RE5djcxK-97dUHzPWY7GEhI8/edit?usp=sharing

The document contains:

1. annotation of stance:

sheet: mask, incubation, asymtomatic, seasonality, column 'stance'

2. annotation for relevance

sheet: mask, incubation, asymtomatic, seasonality, column 'relevance'

3. annotation for system evaluation 

sheet: system_eval_varname, column 'relevance'


### Evaluation of the system:
For evaluation of the system, we first use keyword approach to extract abstract contains the keywords. Then we mannually annotate whether the abstract extracted are relevant to the question asked. We compute the precision and recall of our system based on this annotation 



### file documentation

* test_collection.py: create test collection by extracting articles contain specific keywords in the abstract

* preprocess.py: read metadata file and preprocessing data for LDA. Preprocessing include selecting abstract contain certain keywords, stemming, tokenization and extract entities


























