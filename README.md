# Fighting COVID-19 Infodemics 
## Project Design

The knowledge gap between the public and specialists and uncertainties are important factors that drive pandemic anxiety. The general public mainly gains information about the pandemic from social media platforms or simply by a search engine. Information from reliable sources is often ranked at the top by search engines. However, at the beginning of a pandemic, the government and institutes have little information about the virus. Therefore, guidance about protection measures and information about the virus was confusing. The public would benefit from retrieving information from academic papers. However, searching for academic papers is not a trivial task for someone without domain-specific knowledge. It is also an intimidating task for them to summarize the result by reading through dozens of papers.

Here I develop a search system to extract sentences from abstracts that are relevant to a COVID question. The example questions we used here are associated with rumor and uncertain information circulating in public. 

Q1: Is wearing a mask an effective means to control the community spread of the pandemic COVID-19? Governments from Asian countries enforce mask-wearing in public areas and believe this is an effective way to reduce infection risk. By contrast, governments from Europe and U.S. advocate that there is little evidence that wearing a mask is effective in controlling the pandemic, and healthy people in those countries are reluctant to wear masks in their everyday lives. 

Q2: Is 14-day's self-quarantine period enough to ensure no infectiousness of the people quarantined? This question concerns the incubation period of COVID-19. Many governments worldwide use 14 days as the maximum self-quarantine period to check if anyone has a visiting history to the outbreak places or a contracting history with identified patients. However, rumors say that the incubation period of COVID-19 could be longer than 14 days in some social media. 

Q3: Are asymptomatic patients infectious to others? This is the most relevant question to pandemic anxiety. Clarifying this question is critical to the general public health and decisions on whether conducting the virus check only on people showing symptoms or on the whole population (if possible)

Q4: Is the spread of COVID-19 seasonally sensitive?} There is a widely spread rumor claiming that the virus will disappear in the summer, like previous coronavirus outbreaks like SARS or MERS. Clarifying this issue can avoid relaxing the vigilance of the virus too early. 


## Methods:
###  The search system:
The search system first extracts abstract contains a keyword (e.g. 'mask'), then we use LDA to group the abstract topics. The input of the LDA topic modeling are nouns, verbs and adjectives of the abstract. Next, we identify the most relevant topic to the question we asked and extract the abstracts that contain the target topic. Later, the system retrieves sentences that contain the keyword from the selected abstracts. Sentences retrieved from one abstract are grouped as a document. Finally, we use TFIDF to rank these documents.

The process can be described as:

(1) keyword searching abstracts -> (2) extract abstract topics with LDA -> (3) select abstracts with relevant topics -> (3) extract key sentences from selected abstracts -> (4) rank the key sentences.

Users can customize their search in step 3. For example, users want to identify articles for a specific topic (e.g., covid-19). In this case, users are opting to add title filer for different queries in our system.

This approach's benefit is that when we are not familiar with the jargons in an academic field, the search system returns topics (step 3) that primes you for a more accurate search.



### System Evaluation:
We manually annotate the key sentences to identify the result sentences' stance and whether these sentences are relevant to the question asked.


Keywords:
Incubation period, asymptomatic, mask, death rate, paracetamol


### Annotation
To understand the answer to the relevant question, we need to annotate the results' stance, such as, does the abstract for/against the statement. 

To evaluate the search system, we need to annotate the relevance of the retrieved result. Please refer to each section for annotation guideline.

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
For evaluation of the system, we first use the keyword approach to extract abstract contains the keywords. Then we manually annotate whether the abstract extracted are relevant to the question asked. We compute the precision and recall of our system based on this annotation. 



### file documentation

* test_collection.py: create test collection by extracting articles contain specific keywords in the abstract

* preprocess.py: read metadata file and preprocessing data for LDA. Preprocessing include selecting abstract contain certain keywords, stemming, tokenization and extract entities





























