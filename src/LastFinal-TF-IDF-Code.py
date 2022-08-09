import os
import io

import nltk
from nltk.corpus import stopwords
import pandas as pd
#from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
import re
from nltk.tokenize import word_tokenize
#from nltk.tokenize import sent_tokenize
#import pandas as pd
#from string import punctuation
import  numpy as np
from nltk.corpus import stopwords
import gensim
from gensim import corpora
import statistics
import csv

work_dir = "C:/Users/intel/Desktop/testtest"
DIR = 'C:/Users/intel/Desktop/testtest'
## getting the number of files inside the folder as defined file_count variable
file_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

list_name = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

######
firstListOfFilel = []
scount = 0
_words = []
for v in range(len(list_name)):
    sp = re.sub("[_?].*[.?]", "", list_name[v])
    _words.append(sp.lower())
######
appending_listOfSum = []
listOfSim = []
Total_List_Of_sim = []
list2 = []
proba_list =[]
###############Start of block separation

for index in range(0, file_count):
    name = list_name[index]
    print(name)
    path = os.path.join(work_dir, name)
    with io.open(path, mode="r", encoding="utf-8") as fd:
        content = fd.read()
    
    #############END
    
    s = re.split("##", content.strip())
    # print(s)
    rx = re.compile(r'(?<=[a-z])(?=[A-Z])')
    
    nstrings = [rx.sub('#', ll) for ll in s]
    print("nstrings ::", nstrings)
    print(len(nstrings))
    
    if len(nstrings) >= 2 and nstrings != '':
        scount = scount + 1
        
        ##########
        sp = re.sub("[_?].*[.?]", "", name)
        name11 = (sp.lower())
        #########
        words = [w.split("#") for w in nstrings]
        print("WORDS ::: ", words)
        file_lst_trimmed = []
        for p in nstrings:
            file_lst_trimmed.append(re.sub(r'#', ' ', p))
        
        ##############END
        print("file_lst_trimmed:: ", file_lst_trimmed)
        
        ###Tokenizing the blocks
        final_list = [re.sub(' ', '#', i).lower() for i in nstrings]
        x = [re.sub(r'[^A-Za-z0-9]+', ' ', x) for x in final_list]
        # x = [re.sub(r'', '#', x) for x in words]
        # print("\n XXXX: ", x)
        #######
        x1 = [re.sub(r'[/0-9/]+', '', x1) for x1 in x]
        
        new_list1 = list(filter(None, x1))
        
        #######
        matrix = [line.split() for line in new_list1]
        # print("\nMatrix Values : ", matrix, "\n")
        
        stop_words = set(stopwords.words('english'))
        newStopWords = ['.', ';', '{', '}', ']', '[', '//', '(', ')', "/", '=', '#', '*', '0']
        new_stopwords_list = stop_words.union(newStopWords)
        # print(len(matrix))
        word_stemmer = nltk.WordNetLemmatizer()
        TokensWOStop = []
        for item in matrix:
            temp = []
            for word in item:
                if word not in new_stopwords_list and len(word) > 2 and word not in temp and word != '':
                    temp.append(word)
            if len(temp) >= 2:
                TokensWOStop.append(temp)
        
        print("TokensWOStop_Value: ", TokensWOStop)
        print("Len of TokensWOStop_Value: ", len(TokensWOStop))
    ########

    ## cleaning extra stopwords like as numbers and ....
    # tokenized_issue = [[re.sub(r'[^A-Za-z0-9]+', ' ', item.split()) for item in inner_list][0] for inner_list in TokensWOStop]
    # add matrix of values into textfile.txt
    # print("tokenized_issue",tokenized_issue)
    with open('textfile.txt', 'w') as testfile:
        
        for row in TokensWOStop:
            
            testfile.write(' '.join([str(a).lower() for a in row]) + '\n')
            
            print("len of row: ", len(row))
        testfile.close()
    #####################
   
    #####################
    mlb = MultiLabelBinarizer()
    # data = [['public', 'static', 'void', 'main(String', 'args[])', '{','a','1'],['public','3','two', 'matrices', 'int','and','void']]
    #print("mlb.fit_transform(matrix) \n", mlb.fit_transform(TokensWOStop))
    #print("Number of documents and Blocks:", len(TokensWOStop), "\n")

    dictionary = gensim.corpora.Dictionary(TokensWOStop)
    print(dictionary.token2id, "\n")
    # Create a bag of words
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in TokensWOStop]
    #print("corpus : ", corpus)
    # words that occur more frequently across the documents get smaller weights.
    tf_idf = gensim.models.TfidfModel(corpus)
    for doc in tf_idf[corpus]:
        print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

    # Creating similarity measure object
    # building the index
    sims = gensim.similarities.Similarity('workdir/', tf_idf[corpus],
                                          num_features=len(dictionary))
    print("\n Sims \n", sims)
    #
    ###how similar is this query document to each document in the index

    file2_docs = []

    with open('textfile.txt') as f:
    
        lines = f.readlines()
        for line in lines:
            lines = f.read().split("\n")
            file2_docs.append(line)
    
        f.close()
    #print("\n line: ", file2_docs)
    # print("\n list of file2 :",file2_docs)

    ############

    # print("\n","Number of documents in file11111:", len(TokensWOStop),"\n")
    print("Number of documents in file22222:", len(file2_docs), "\n")
   
    # start of sumavg fot comparing Docs To Docs
    avg_list = []
    avg_sims = []  # array of averages
    
    n = len(file2_docs)
    total_sum = []
    sim_value=[]
    mm=0
    # for line in query documents
    for line in (file2_docs):
        # tokenize words
        query_doc = [w.lower() for w in word_tokenize(line)]
        
        # create bag of words
        query_doc_bow = dictionary.doc2bow(query_doc)
        # find similarity for each document
        query_doc_tf_idf = tf_idf[query_doc_bow]
        # print (document_number, document_similarity)
        print('Comparing Result:', sims[query_doc_tf_idf])
        new_x = np.delete(sims[query_doc_tf_idf], mm)
        print("new_x ::",new_x)
        sum_of_sims = (np.sum(new_x))
        print("sum_of_sims ::",sum_of_sims)
        sim_value.append(sum_of_sims/len(new_x))
        mm = mm+1
    # round the value and multiply by 100 to format it as percentage
    print("sim_value",sim_value)
    
   
    
    
    avg_sum_sim_value = (sum_of_sims)/n
    print("avg_sum_sim_value  : ", avg_sum_sim_value)
    print("MethodCohesion : ", avg_sum_sim_value)
    
   
    '''if avg_sum_sim_value<0 :
        avg_sum_sim_value = 1
    elif avg_sum_sim_value>1 :
        avg_sum_sim_value = 0'''
    PLM = 1 - avg_sum_sim_value
    print("1-MethodCohesion (PLM) --Distance:", PLM)
    proba_list.append(PLM)
    firstListOfFilel.append(name11)
    
print("list of proba'z : ",proba_list)

# get the median of proba_list
print("the median of proba'z : ",statistics.median(proba_list))
## import values in CSV file
## Start of making smell_list of smelled method name

##########
v = open('all_smells11.csv')
r = csv.reader(v)
row0 = next(r)
row0.append('y_act')
# print (row0)
actual_list = []
i = 0
count = 0
with open('outfile_prob133.csv', 'w', encoding="utf-8", newline='') as myfile:
    wr = csv.writer(myfile, dialect='excel')
    wr.writerow(("method_name", "y_act"))
    for item in r:
        if item[4] == "Long Method":
            ss = (item[2] + item[3])
            # print(ss.lower())
            wr.writerow(((ss.lower()), '1'))
            count = count + 1
        
        # print (item[0])

myfile.close()
##########
list_of_smell_method = []
acount = 0
with open('outfile_prob133.csv', 'r') as fff:
    reader1 = csv.reader(fff)
    print("true")
    for row in reader1:
        # acount = acount+1
        list_of_smell_method.append(row[0])

## compare the file method with smell list of methods to retrieving the actual list to use
y_actt = []
for t in range(len(firstListOfFilel)):
    if firstListOfFilel[t] in list_of_smell_method:
        y_actt.append('1')
    elif firstListOfFilel[t] not in list_of_smell_method:
        y_actt.append('0')
    
    elif list_of_smell_method[t] not in firstListOfFilel:
        print("false")


print("y_actt: ", y_actt)
print("Len(y_actt) :: ", len(y_actt))

y_act_count = 0
for m in range(len(y_actt)):
    if y_actt[m] == '1':
        y_act_count = y_act_count + 1
print("y_act_count :", y_act_count)
print("proba_list :", Total_List_Of_sim)
print("list of method_names ::: ",firstListOfFilel)
## end of making csv fil
#y_act=[1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,1]

#############
print("list_of_method_with value (1) :: ", list_of_smell_method)
print("Len_ list_of_method_with value (1) :: ", len(list_of_smell_method))
### start maching the actual file
with open('outfile_prob1.csv', 'w', encoding="utf-8", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("y_act", "y_pred", "y_name"))
    
    wr.writerows(zip(y_actt, proba_list, firstListOfFilel))

import pandas as pd

### Start the code of measurments and evaluation
file_path = 'outfile_prob1.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)
print(df.head(20000))