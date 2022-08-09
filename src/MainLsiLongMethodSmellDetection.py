# PACKAGES
import nltk
import re
import statistics
import numpy as np
import csv
from gensim import corpora, similarities
from gensim.models import LsiModel
from nltk.corpus import stopwords
# nltk.download("stopwords")
# nltk.download("wordnet")
import os
import io
import math

work_dir = "../Longmethod_Blocks"
DIR = '../Longmethod_Blocks'
## getting the number of files inside the folder as defined file_count variable
file_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

list_name = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

######
appending_listOf_Newx = []
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
    print(len(nstrings)-1)
    
    if len(nstrings) <= 5 and nstrings != '':
        scount = scount + 1
        
        ##########
        sp = re.sub("[_?].*[.?]", "", name)
        name11 = (sp.lower())
        #########
       
        # FUNCTION of calling for tokenizing and stemming the blocks befor the comparison
        def filter_words_and_get_word_stems(document, word_tokenizer, word_stemmer,
                                            stopword_set, pattern_to_match_words=r"[^\w]",
                                            word_length_minimum_n_chars=2):
            """Remove multiple white spaces and all non word content from text and
            extract words. Then filter out stopwords and words with a length smaller
            than word_length_minimum and apply word stemmer to get wordstems. Finally
            return word stems.
            """
            document = re.sub(pattern_to_match_words, r" ", document)
            document = re.sub(r"\s+", r" ", document)
            words = word_tokenizer.tokenize(document)
            words_filtered = [word.lower()
                              for word in words
                              if word.lower() not in stopword_set and len(word) >= word_length_minimum_n_chars]
            word_stems = [word_stemmer.lemmatize(word) for word in words_filtered]
            return (word_stems)
        
        
        '''
        # training text data to calculate TF-IDF model from
        documents_train = ['void#setCvsRoot#(#String#root#)#{', '// Check if not real cvsroot => set it to null',
                           'if#(#root#.#trim#(#)#.#equals#(#""#)#)#{#root#=#null#;#}#}']
    
        # test data text data to match
        document_test = '// Check if not real cvsroot => set it to null'
        '''
        # PREPROCESS
        
        # set stopword set, word stemmer and word tokenizer
        stop_words = set(stopwords.words('english'))
        newStopWords = ['.', ';', '{', '}', ']', '[', '//', '(', ')', "/", '=', '#', '*', '0']
        new_stopwords_list = stop_words.union(newStopWords)
        word_tokenizer = nltk.tokenize.WordPunctTokenizer()
        word_stemmer = nltk.WordNetLemmatizer()

        # Start the blocks comparison (1-2)(1-3),....
        ss = []

        for kk in range(len(nstrings)):
            if nstrings[kk] != '':
                ss.append(nstrings[kk])
        if len(ss) > 1:
          for m in range(len(ss)):
          
              print("mmm :::",m)
            
              # print("SS:: after DELETE", ss)
              ############
            
              ############
              word_stem_arrays_train = [
                filter_words_and_get_word_stems(
                    str(document),
                    word_tokenizer,
                    word_stemmer,
                    new_stopwords_list
                ) for document in ss]
            
            # apply cleaning, filtering and word stemming to test document
            
              word_stem_array_test = filter_words_and_get_word_stems(
                str(ss[m]),
                word_tokenizer,
                word_stemmer,
                new_stopwords_list)
            
              # PROCESS
            
            # create dictionary containing unique word stems of training documents
            # TF (term frequencies) or "global weights"
              dictionary = corpora.Dictionary(
                word_stem_array_train
                for word_stem_array_train in word_stem_arrays_train)
              # print("Dictionary :", dictionary)
            
            # create corpus containing word stem id from dictionary and word stem count
            # for each word in each document
            # DF (document frequencies, for all terms in each document) or "local weights"
              corpus = [
                dictionary.doc2bow(word_stem_array_train)
                for word_stem_array_train in word_stem_arrays_train]
              print("Corpus :", corpus)
            
              # create LSI model (Latent Semantic Indexing) from corpus and dictionary
            # LSI model consists of Singular Value Decomposition (SVD) of
            # Term Document Matrix M: M = T x S x D'
            # and dimensionality reductions of T, S and D ("Derivation")
              lsi_model = LsiModel(
                corpus=corpus,
                id2word=dictionary  # , num_topics = 2 #(opt. setting for explicit dim. change)
              )
              # print("Derivation of Term Matrix T of Training Document Word Stems: ", lsi_model.get_topics())
            
            # Derivation of Term Document Matrix of Training Document Word Stems = M' x [Derivation of T]
            # print("LSI Vectors of Training Document Word Stems: ",[lsi_model[document_word_stems] for document_word_stems in corpus])
            
            # calculate cosine similarity matrix for all training document LSI vectors
              cosine_similarity_matrix = similarities.MatrixSimilarity(lsi_model[corpus])
              print("Cosine Similarities of LSI Vectors of Training Documents:",[row for row in cosine_similarity_matrix])
            
            #
            # calculate LSI vector from word stem counts of the test document and the LSI model content
              vector_lsi_test = lsi_model[dictionary.doc2bow(word_stem_array_test)]
              # print("LSI Vector Test Document:", vector_lsi_test)
            
            # perform a similarity query against the corpus
              cosine_similarities_test = cosine_similarity_matrix[vector_lsi_test]
              print("Cosine Similarities of Test Document LSI Vectors to Training Documents LSI Vectors:",
                  cosine_similarities_test)

            # end of the cosine_similarities for a block
            ############
              new_x = []
              print("Null of New_x :", new_x)
            
              if len(ss) > 1:
    
                m = m + 1
                
                new_x = cosine_similarities_test
                print("cosine_similarities_test :: ",new_x)
                C = m
                for H in range(m):
                    new_x = np.delete(new_x, C-1)
                    
                    C=C-1
              else:
                    new_x = cosine_similarities_test
            
             
              print("new_x after del :: ", new_x)
              for B in range(len(new_x)):
                  
                  appending_listOf_Newx.append(new_x[B])
              print("listOfSum :: ", appending_listOf_Newx)
            
          
          # OUTPUT
        
        # get text of test documents most similar training document
        # most_similar_document_test = documents_train[np.argmax(cosine_similarities_test)]
        # print("Most similar Training Document to Test Document:", most_similar_document_test)
        
        # Sum and average and PLM variables to create the proba'z
          for w in range(len(appending_listOf_Newx)):
              if 1 <= appending_listOf_Newx[w] or appending_listOf_Newx[w] <= 0:
                  print(appending_listOf_Newx[w])
                  xz = 1 / (1 + math.exp(-(appending_listOf_Newx[w])))
                  # print((xz))
                  appending_listOf_Newx[w] = xz
          listOfSum = 0
          print(type(new_x))
          print("appending_listOfSum::", appending_listOf_Newx)
          print("len_of_cosine_sim_test ::", len(cosine_similarities_test))
          v = len(appending_listOf_Newx)
        
          print("mean of listOfSim ::", sum(appending_listOf_Newx) / v)
          simValue = 1 - (sum(appending_listOf_Newx) / v)
          if 1 <= simValue or simValue <= 0:
              xzd = 1 / (1 + math.exp(-(simValue)))
            
              Total_List_Of_sim.append(xzd)
          else:
              Total_List_Of_sim.append(simValue)
        # print("simValue ::: ",simValue)
        
          print("Total_List_Of_sim : ", Total_List_Of_sim)
          firstListOfFilel.append(name11)
        
          appending_listOf_Newx = []

print("Total_List_Of_sim :: ", Total_List_Of_sim)

thresh = statistics.median(Total_List_Of_sim)
print("the threshold of prob'z : ", thresh)
print("len of all probabilities", len(Total_List_Of_sim))

######### Input retrieved method in a list

with open('outfile_prob_of_method.csv', 'w') as f:
    w = csv.writer(f)
    for row in zip(firstListOfFilel):
        w.writerow(row)
#######

#########END of

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

## end of making csv fil


##start of making a list of maching (text_files with prob11.csv)
'''

actual_list = []
list_nested = []
NotEqualFiles = []
ii = 0
z = 0
with open('outfile_prob11.csv', 'rt') as f:
    reader = csv.reader(f)
    # print(n)
    print("it's loading ...")
    for n in range(scount):
        next(reader)
        H = 0
        # print(n)
        for row in reader:

            if _words[n] == row[0]:
                H = H + 1
                if len(list_nested) > 0 and list_nested[0] == row[0]:

                    if list_nested[1] == '0' and row[1] == '1':
                        z = z + 1
                        list_nested = []
                        list_nested.append(_words[n])
                        list_nested.append(row[1])

                        # print(list_nested[0])
                        # print(row[0])
                        # print(row[1])

                        del actual_list[ii - z]

                        # print(actual_list)
                        actual_list.append(list_nested)
                        # print(actual_list)
                        ii = ii + 1
                        # print("iiiii : ", ii)

                else:
                    list_nested = []
                    list_nested.append(_words[n])
                    list_nested.append(row[1])

                    actual_list.append(list_nested)

                    ii = ii + 1
                    # print("ii:::", ii)

        if H == 0:
            NotEqualFiles.append(_words[n])

        f.seek(0)

f.close()

######
## end of list of maching as atual_list

# making the list of prob11.csv as temp2 list name
######
temp2 = []

with open('outfile_prob122.csv', mode='r') as ff:
    reader = csv.reader(ff)
    for k, row in enumerate(reader):
        if not k:  # skip header
            continue

        temp2.append(row[1])

###### End of making list of temp2

## start of making y_act and method name list
act_var = []
act_var_number = []
for iii in range(len(actual_list)):
    act_list_cover = []
    act_list_cover = (actual_list[iii])

    act_var.append(act_list_cover[0])
    act_var_number.append(act_list_cover[1])

print("act_var : ",act_var)
print("act_var_number ",act_var_number)
print("actual_list:: ",actual_list)
########
kk = 0
y_act = []
cover_list = []
for item in temp2:

    # cover_list.append (item)
    # cover_list=(actual_list[kk])
    # print(cover_list)
    if item in act_var:
        for LL in range(len(act_var)):
            if item == act_var[LL]:
                kk = act_var.index(act_var[LL])

        # temp5.append(item)
        # print("kk:: ", kk)
        y_act.append(int(act_var_number[kk]))

    # y_act.append(row[0])
    else:
        y_act.append(0)


'''

##end of making or retrieving The y_act list
#################
# start of evaluation of probz
print("y_actt: ", y_actt)
print("Len(y_actt) :: ", len(y_actt))

y_act_count = 0
for m in range(len(y_actt)):
    if y_actt[m] == '1':
        y_act_count = y_act_count + 1
print("y_act_count :", y_act_count)
print("proba_list :", Total_List_Of_sim)
print("list of method_names ::: ", firstListOfFilel)

# print("thlist ::::: ", thlist,'/n')
print("number of blocks: ", scount)

#############
print("list_of_method_with value (1) :: ", list_of_smell_method)
print("Len_ list_of_method_with value (1) :: ", len(list_of_smell_method))
### start maching the actual file
with open('outfile_prob1.csv', 'w', encoding="utf-8", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(("y_act", "y_pred", "y_name"))
    
    wr.writerows(zip(y_actt, Total_List_Of_sim, firstListOfFilel))

import pandas as pd

### Start the code of measurments and evaluation
file_path = 'outfile_prob1.csv'
data = pd.read_csv(file_path)
df = pd.DataFrame(data)
print(df.head(20000))


## get the threshold by prob'z median
def the_evaluation(thresh):
    ########
    thcount = 0
    List_Of_smells = []
    List_Of_smelled_method = []
    for jj in range(len(Total_List_Of_sim)):
        if Total_List_Of_sim[jj] > thresh:
            thcount = thcount + 1
            List_Of_smells.append(Total_List_Of_sim[jj])
            List_Of_smelled_method.append(firstListOfFilel[jj])
    
    print("list of detected smell as prob'z vlaue:: ", List_Of_smells)
    print("list of Name detected smell as block name:: ", List_Of_smelled_method)
    print("thcount :: ", thcount)
    ########
    ########y_act_count
    file_path = 'outfile_prob1.csv'
    data = pd.read_csv(file_path)
    df = pd.DataFrame(data)
    df.head(20000)
    df['y_pred_rf'] = (df.y_pred >= thresh).astype('int')
    print(df.head(20000))
    df.shape
    
    def compute_tp_tn_fn_fp(y_act, y_pred):
        tp = sum((y_act == 1) & (y_pred == 1))
        tn = sum((y_act == 0) & (y_pred == 0))
        fn = sum((y_act == 1) & (y_pred == 0))
        fp = sum((y_act == 0) & (y_pred == 1))
        
        return tp, tn, fp, fn
    
    tp_lr, tn_lr, fp_lr, fn_lr = compute_tp_tn_fn_fp(df.y_act, df.y_pred_rf)
    print('TP for being smell :', tp_lr)
    print('TN for being smell :', tn_lr)
    print('FP for being smell :', fp_lr)
    print('FN for being smell :', fn_lr)
    
    from sklearn.metrics import accuracy_score
    
    accuracy = 100 * accuracy_score(df.y_act, df.y_pred_rf)
    # print('Accuracy for being smell :', 100 * accuracy_score(df.y_act, df.y_pred_rf))
    list_of_evaluations = []
    list_of_evaluations.append(accuracy)
    
    from sklearn.metrics import precision_score
    
    precision_sc = 100 * precision_score(df.y_act, df.y_pred_rf)
    # print('precision for being smell : ', 100 * precision_score(df.y_act, df.y_pred_rf))
    list_of_evaluations.append(precision_sc)
    
    from sklearn.metrics import recall_score
    
    recall_sc = 100 * recall_score(df.y_act, df.y_pred_rf)
    # print('Recall for being smell :', 100 * recall_score(df.y_act, df.y_pred_rf))
    list_of_evaluations.append(recall_sc)
    
    from sklearn.metrics import f1_score
    
    f1_sc = 100 * f1_score(df.y_act, df.y_pred_rf)
    # print('F1_score for being smell : ', f1_score(df.y_act, df.y_pred_rf))
    list_of_evaluations.append(f1_sc)
    # print([list_of_evaluations], "\n")
    
    h = ['Accuracy', 'Precision', 'Recall', 'F1_score']
    print('{:<9s} {:<10s} {:<8s} {:<15s}'.format(*h))
    for list_ in [list_of_evaluations]:
        print('{:.<2f} {:.<2f} {:.<2f} {:.<2f}'.format(*list_))


# End of evaluationz
the_evaluation(thresh)
print(scount)

########
while True:
    value = input("Please enter new threshold value then press Enter key :\n")
    value = float(value)
    print(f'You entered {value} and the Result is :')
    print(the_evaluation(value))
    print('the new value of threshold is : ', value)