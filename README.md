# Fighting COVID-19 Infodemics 
## Project Design

Knowledge gap between public and specialists and uncertainties are an important factors that drive pandemic anxiety. The general public mainly gain information about the pandmic from social media platforms or simply by search engine. Information from reliable sources is often ranked at the top by search engines. However, at the begining of a pandemic, government and institutes have little information about the virus, therefore, guidance about protect measure and information about the virus was confusing. The public would benefit from retrieving information from academic papers. However, searching information from academic paper is not a trivial task for someone without a domain specific knowledge. It is also an intimidating task for someone without domain knowledge to summarize the result by reading through dozens of papers.

Here I develop a search system to extract sentences from abstracts that are relevant to a COVID question. The example questions we used in here are associated with rumour and uncertain information circulating in the public. 

Q1: Is wearing mask an effective means to control community spread of the pandemic COVID-19? Governments from Asian countries enforce mask wearing in public areas and believe this is an effective way to reduce the risk of infection. By contrast, governments from Europe and U.S. advocate that there is little evidence that wearing mask is effective in controlling the pandemic, and healthy people in those countries are reluctant to wear masks in their everyday lives. 

Q2: Is 14-day's self-quarantine period enough to ensure no infectiousness of the people quarantined? This question concerns the incubation period of COVID-19. Many governments in the world use 14 days as the maximum self-quarantine period to check if anyone who has a visiting history to the outbreak places or a contacting history with identified patients. However, in some social media, there are rumors saying that the incubation period of COVID-19 could be longer than 14 days. 

Q3: Are asymptomatic patients infectious to others? This is the most relevant question to the pandemic anxiety. Clarifying this question is critical to the general public health and decisions on whether conducting the virus check only on people showing symptoms or on the whole population (if possible)

Q4: Is the spread of COVID-19 seasonally sensitive?} There is a widely spread rumor claiming that the virus will disappear in the summer, like previous coronavirus outbreaks like SARS or MERS. Clarifying this issue can avoid relaxing the vigilance on the virus too early. 


## Methods:
###  The search system:
The search system first extract abstract contains a keyword (e.g. ‘mask’), then we use LDA to group the abstract topics. The input of the LDA topic modeling are nouns, verbs and adjectives of the abstract. Next, we identify a topic that is most relevant to the question we asked and we extract the abstracts that contain the target topic. Later, the system retrieves sentences that contain the keyword from the selected abstracts. Sentences retrieved from one abstract are grouped as a document. Finally, we use TFIDF to rank these documents.

The process can be described as:

(1) keyword searching abstracts -> (2) extract abstract topics with LDA -> (3) select abstracts with relevant topics -> (3) extract key sentences from selected abstracts -> (4) rank the key sentences.

Users can custumize their search in step 3. For some queries, users want to identify articles for a specific topic (e.g. covid-19). In this case users are opt to add title filer for different queries in our system.

The benefit of this approach is that when we are not familar with the jagons in an academic field, the search system returns topics (step 3) that primes you for a more accurate search.



### System Evaluation:
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


























