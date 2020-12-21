#Data Frame Creation
import pandas as pd
#file reading
import glob
import json
import random
random.seed(4)
#Preprocess Method
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.porter import PorterStemmer
from nltk.tokenize import word_tokenize
#text cleaning methods
import string
import re
#LDA Topic Modeling
from scipy.spatial import distance
from gensim import models, matutils
import gensim.corpora as corpora
#formatting output
from pprint import pprint

#A helper method to help clean strings
def clean_text(text):
    '''Make text lowercase, remove parenthses, remove
    punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('[(-\\\/)]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub("\s\s+", " ", text)
    text = re.sub('\n', '', text)
    return text
    
#Retrieving the file paths of all articles and randomly picking a subset
filepaths = glob.glob("../input/CORD-19-research-challenge/document_parses/pdf_json/*.json" ,recursive = True)
articles = random.sample(filepaths,2000) #taking a sample as the corpus is
print(articles)

#Extract Json data into Pandas Data Frame
root_path = '/kaggle/input/CORD-19-research-challenge/document-parses'
stop_words = STOPWORDS #You may add your own custom Stop Words as well
dict_ = {'paper_id': [],'content': []}
i=0 #counter to keep track of progress
print('Starting Now ...')
for article in articles:
    #print out progress(change 500 with regard to len(articles))
    if(i%500 == 0):
        print(str(i)+": Document Checkpoint")
    i= i + 1
    with open(article) as json_file:
        
        #retrieving the data from content
        raw_json_data = json.load(json_file)
        paper_id = raw_json_data["paper_id"]
        title = raw_json_data["metadata"]["title"]
        body_list = raw_json_data["body_text"]
        body = ''
        #"body_text" is a list of strings so they must be
        #concatanated together with a seperating space
        for paragraph in body_list:
            body = body + str(paragraph['text'])
        content = title+ " " + body
        
        #cleaning, stemming, and tokenizing the content
        content = clean_text(content)
        content = PorterStemmer().stem_sentence(content)
        tokens = word_tokenize(content)
        
        #adding the entry to the dictionary
        dict_['paper_id'].append(paper_id)
        dict_['content'].append([word for word in tokens if not word in stop_words])
        
#creaing Pandas Dataframe
df_covid = pd.DataFrame(dict_, columns=['paper_id','content'])
df_covid.head()

# Create Dictionary
id2word = corpora.Dictionary(df_covid['content'])
print("created dictionary")
print(id2word)
# Create Corpus
texts = df_covid['content']

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
print('created corpus')
# View
print(corpus[:1])

#LDA high-topic model
ldah = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=150, passes=10)
print("training finshed")

#the prompt that was given for each general question along with related articles
risk_factors_prompt = '''What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?
Specifically, we want to know what the literature reports about:
Data on potential risks factors
Smoking, pre-existing pulmonary disease
Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
Neonates and pregnant women
Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
Susceptibility of populations
Public health mitigation measures that could be effective for control '''

therapeutics_prompt = '''What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?
Specifically, we want to know what the literature reports about:
Effectiveness of drugs being developed and tried to treat COVID-19 patients.
Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.
Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.
Exploration of use of best animal models and their predictive value for a human vaccine.
Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.
Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.
Efforts targeted at a universal coronavirus vaccine.
Efforts to develop animal models and standardize challenge studies
Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers
Approaches to evaluate risk for enhanced disease after vaccination
Essays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]'''

diagnostics_prompt = ''' What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?
Specifically, we want to know what the literature reports about:
How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).
Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.
Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.
National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).
Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.
Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).
Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.
Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.
Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.
Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.
Policies and protocols for screening and testing.
Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.
Technology roadmap for diagnostics.
Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.
New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.
Coupling genomics and diagnostic testing on a large scale.
Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.
Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.
One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors.'''

#collecting the names of all the studies of all articles included
def collate_titles(filepaths):
    titles = ''
    for csv in filepaths:
        raw_csv_data = pd.read_csv(csv)
        for index,row in raw_csv_data.iterrows():
            titles = titles + ' ' + row['Study']
    return titles

#file paths of relevant articles
therapeutics_filepaths = glob.glob("../input/CORD-19-research-challenge/Kaggle/target_tables/7_therapeutics_interventions_and_clinical_studies/*.csv" ,recursive = True)
diagnostics_filepaths = glob.glob("../input/CORD-19-research-challenge/Kaggle/target_tables/8_risk_factors/*.csv" ,recursive = True)
risk_factors_filepaths = glob.glob("../input/CORD-19-research-challenge/Kaggle/target_tables/6_diagnostics/*.csv" ,recursive = True)

#collected titles of each prompt
risk_factors_titles = collate_titles(risk_factors_filepaths)
diagnostics_titles = collate_titles(diagnostics_filepaths)
therapeutics_titles = collate_titles(therapeutics_filepaths)

#the full string of relevant information for each prompt
risk_factors = risk_factors_prompt + ' ' + risk_factors_titles
diagnostics = diagnostics_prompt + ' ' + diagnostics_titles
therapeutics = therapeutics_prompt + ' ' + therapeutics_titles

#A helper method that returns the topic distribution for the given text
def get_topic_dist(ldah, texts):
    corpus = id2word.doc2bow(texts)
    return ldah.get_document_topics(corpus)

doc_dists = [] #holds the topic distribution for every article
i = 0 #counter
for index, row in df_covid.iterrows():
    
    #checkpoint generator
    if i % (len(df_covid) // 10) == 0:
        print(f'Processing index: {i} of {len(df_covid)}')
    i = i + 1
    
    #compute topic distribution
    dist = get_topic_dist(ldah, row['content'])
    #sort the distribution in an increasing order of topic number
    dist.sort()
    doc_dists.append(dist)
    
#adding the topic distributions to the dataframe
df_covid['topic_dist'] = doc_dists
print(df_covid.head())

#compute topic distribution on tasks/prompts
def get_task_topic_dist(ldah, task):
    #clean, stem and tokenize the prompt/task string
    task = clean_text(task)
    task = PorterStemmer().stem_sentence(task)
    tokens = word_tokenize(task)
    tokens = [word for word in tokens if not word in stop_words]
    
    #compute topic distribution and sort
    dist = get_topic_dist(ldah, tokens)
    dist.sort()
    
    return dist

#compute topic distributions
diagnostics_dist = get_task_topic_dist(ldah, diagnostics)
therapeutics_dist = get_task_topic_dist(ldah, therapeutics)
risk_factors_dist = get_task_topic_dist(ldah, risk_factors)

#check out the distributions
print(diagnostics_dist)
print(therapeutics_dist)
print(risk_factors_dist)

#create a 1-D Vector of topic probabilities with size of len(num_of_topics)
def create_full_vectors(dist1):
    i = 0 #topic number
    j = 0 #index of dist1
    dist1_final = [] #initalizing final vector
    
    #while i < num_of_topics
    while(i < 150):
        
        #check if j is a valid index and if the current position in dist1 has a topic number = i
        if(j < len(dist1) and dist1[j][0] == i):
            dist1_final.append(dist1[j][1])
            j = j+1
            
        #else just add a probability of 0
        else:
            dist1_final.append(0.0)
            
        i = i + 1
        
    return dist1_final

#creating the column in the data frame for a full 1-D probability vector for comparasion purposes
full_doc_dist = []
i=0
for index, row in df_covid.iterrows():
    #progress checkpoint
    if i % 500 == 0:
        print(f'Processing index: {i} of {len(df_covid)}')
    i = i + 1
    dist = row['topic_dist']
    full_doc_dist.append(create_full_vectors(dist))
df_covid['full_topic_dist'] = full_doc_dist
print(df_covid.head())

#create the 1-D vectors for each prompt/task
diagnostics_full_dist = create_full_vectors(diagnostics_dist)
therapeutics_full_dist = create_full_vectors(therapeutics_dist)
risk_factors_full_dist = create_full_vectors(risk_factors_dist)

print(diagnostics_full_dist)

# An example of how not to make a method - for your code only take df and dist as parameters compute jenson-shannon
# distances for one single distribution and call the method three times - or do what I did and get weird with it
def retrieve_related_docs(df,dist1,dist2,dist3):
    dist1_jenson = []
    dist2_jenson = []
    dist3_jenson = []
    for index,row in df.iterrows():
        
        #calculate jenson shannon distances
        diff1 = distance.jensenshannon(dist1, row['full_topic_dist'])
        diff2 = distance.jensenshannon(dist2, row['full_topic_dist'])
        diff3 = distance.jensenshannon(dist3, row['full_topic_dist'])
        
        #append the (difference, paper_id) tuple to the list
        dist1_jenson.append((diff1, row['paper_id']))
        dist2_jenson.append((diff2, row['paper_id']))
        dist3_jenson.append((diff3, row['paper_id']))
    
    #sort based on increasing differences
    dist1_jenson.sort()
    dist2_jenson.sort()
    dist3_jenson.sort()
    
    return dist1_jenson, dist2_jenson, dist3_jenson

#retrieve a list of documents for each task sorted by most similar to least
risk_factors_documents, diagnostics_documents, therapeutics_documents = retrieve_related_docs(df_covid, risk_factors_full_dist, diagnostics_full_dist, therapeutics_full_dist)

pprint(risk_factors_documents[0:5])
print(' ')
pprint(diagnostics_documents[0:5])
print(' ')
pprint(therapeutics_documents[0:5])

#build a dataframe to look up paper_id and retrieve the title of the paper
validation_dict = {'paper_id': [],'title': []}
for article in articles:
    with open(article) as json_file:
        raw_json_data = json.load(json_file)
        paper_id = raw_json_data["paper_id"]
        title = raw_json_data["metadata"]["title"]
        validation_dict['paper_id'].append(paper_id)
        validation_dict['title'].append(title)
df_validation = pd.DataFrame(validation_dict, columns=['paper_id','title'])
print(df_validation)

#DO IT Yourself - build a helper method to facilate title retrieval

#retrieve the title of an article from a cartain spot in the similarity rankings for each topic
print(df_validation.loc[df_validation['paper_id'] == risk_factors_documents[0][1]])
print(df_validation.loc[df_validation['paper_id'] == diagnostics_documents[0][1]])
print(df_validation.loc[df_validation['paper_id'] == therapeutics_documents[0][1]])
