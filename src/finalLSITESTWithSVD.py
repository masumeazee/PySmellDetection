##      ""  This code has been programmed to use LSI technique to find
##          The optimal and efficient way to get inner a method cohesion
##          Implemented with python version 3.7 and anaconda3   ""
##
##          By.Azizyan Masoume

## Most usful ---lib-- as use below
import nltk
import os
import io
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
import re
import math
from numpy.linalg import norm
from numpy import dot
from numpy import array, double
from numpy.linalg import svd
from numpy import zeros
from numpy import diag
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import gensim
from gensim import corpora
from string import punctuation
import statistics
import csv
firstListOfFilel= []
count_name = 0
listoffile = []
scount = 0
proba_list = []
work_dir = "C:/Users/intel/Desktop/testtest"
DIR = 'C:/Users/intel/Desktop/testtest'
## getting the number of files inside the folder as defined file_count variable
file_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

list_name = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
#print(list_name)
## the number of method to get their inner coheiscohesion
## and read from its Folder

## get the lower case of method names from text file
_words = []
for v in range(len(list_name)):
    sp = re.sub("[_?].*[.?]", "", list_name[v])
    _words.append(sp.lower())

print("_text file_names  : ", _words)

## start of making and block separation
list2 = []
for index in range(0, file_count):
    name = list_name[index]
    print(name)
    path = os.path.join(work_dir, name)
    with io.open(path, mode="r", encoding="utf-8") as fd:
        content = fd.read()
    ## inside every method we will have some some blocks(some statements) that already has been splitted
    ## and we call it as a text of some blocks
    s = re.split("##", content.strip())

    rx = re.compile(r'(?<=[a-z])(?=[A-Z])')

    nstrings = [rx.sub('#', ll) for ll in s]
    print("nstrings:: ",nstrings)
    final_list = [re.sub(' ', '#', i).lower() for i in nstrings]
    print("final_list : ",final_list)
    
    #print(xxx)
    
    print(len(final_list))
    if len(final_list) >= 2 and final_list != '':
        scount = scount + 1
        
        ###converting name to be accessable ::::
        
        sp = re.sub("[_?].*[.?]", "", name)
        name11 =(sp.lower())
        # print("The original list is :\n ", s)
        # print("Type of S :\n ", type(s))

        # start file's name separation and to have a total name which combines of class and method name
        
        # start cleaning the Blocks inside methods
        words = [w.split("#") for w in final_list]
        print("words :", words)
        ########
        x = [re.sub(r'[^A-Za-z0-9]+', ' ', x) for x in final_list]
        #x = [re.sub(r'', '#', x) for x in words]
        # print("\n XXXX: ", x)
        #######
        x1 = [re.sub(r'[/0-9/]+', '', x1) for x1 in x]
        
        new_list1 = list(filter(None, x1))
        
        #######
        matrix = [line.split() for line in new_list1]
        #print("\nMatrix Values : ", matrix, "\n")
        
        stop_words = set(stopwords.words('english'))
        newStopWords = ['.', ';', '{', '}', ']', '[', '//', '(', ')', "/", '=', '#', '*', '0']
        new_stopwords_list = stop_words.union(newStopWords)
        #print(len(matrix))
        word_stemmer = nltk.WordNetLemmatizer()
        TokensWOStop = []
        for item in matrix:
            temp = []
            for word in item:
                if word not in new_stopwords_list and len(word) > 2 and word not in temp:
                   temp.append(word)
            TokensWOStop.append(temp)
        
        print("TokensWOStop_Value: ", TokensWOStop)
       
        # print(len(TokensWOStop))
        ########
        ## end of cleaning process
        
        ## here the splitted blocks has been stored inside a text file (line by line)
        ## to use if the process of splitting
    
        with open('textfile.txt', 'w') as testfile:
            
            for row in words:
               testfile.write(' '.join([str(a).lower() for a in row]) + '\n')

                # print("len of row: ", len(row))
            testfile.close()
        #####################
        #####################
        mlb = MultiLabelBinarizer()
        #################
        
        #### show the matrix of Tokens
        #print("mlb.fit_transform(matrix) \n", mlb.fit_transform(TokensWOStop))
        A = np.array(mlb.fit_transform(TokensWOStop))
        #rint(A.T)
        
        #####start calculate SVD to use of reduced matrix
        '''
        U, s, VT = svd(A.T)
        AA = A.T  ## term-document Matrix
        UU = AA.dot(AA.T)
        # print("UU: ",UU)
        # print("UUUU : ",np.dot(AA.T,AA))
        # reciprocals of s
        d = 1.0 / s
        # create m x n D matrix
        D = zeros(A.T.shape)
        # populate D with n x n diagonal matrix
        D[:A.T.shape[1], :A.T.shape[1]] = diag(d)
        # print(diag(d))
        # calculate pseudoinverse
        rows = len(A.T)  # height
        columns = len(A.T[0])  # Width
        k = (rows * columns) ** 0.2  # get K vlaue by the huristic formula of  k =(m*n)^0.2
        #print("k : ", round(k))
        # print("U: \n",np.array(U))
        U_reduced_to_k = U[:, :round(k)]
        #print("U_reduced_to_k: \n", U_reduced_to_k)
        zigma1 = np.diagflat(s)
        zigma_reduced_to_k = zigma1[:round(k), :round(k)]
        # print("S: \n",s)
        # print("S: \n", zigma1)
        #print("S: \n", zigma_reduced_to_k)
        G = VT
        # print("VT: \n",G)
        VT_reduced_to_k = VT[:round(k), :]
        #print("VT_reduced_to_k: \n", VT_reduced_to_k)
        
        FF = VT.T.dot(D.T).dot(U.T)  # final Inverse
        
        ## Strat the comparing of documents (1->2)(1->3)(...)(2->3)(2->4),....
        ## by this formula :: cos_sim = ((x*y)+(z*w))/(math.sqrt((x**2)+(y**2)))*((math.sqrt((x**2)+(y**2))))
        '''
        ###start of SVD
        rows = len(A.T)  # height
        columns = len(A.T[0])  # Width
        k = (rows * columns) ** 0.2  # get K vlaue by the huristic formula of  k =(m*n)^0.2
        #if (k % 1)>0 :
            #kk=round(k-1)
        
        frac, whole = math.modf(k)
        if (whole/1)!=0:
           
           kk=round(k)
        else:
            kk=round(k-1)
            
        print("k : ", kk)
        u, s, vT = np.linalg.svd(A.T)
        print("\n u :\n", u, "\ns :\n", s, "\nvt :\n", vT)
        print("shape(u)::",np.shape(u))
        
        up, sp, vp = u[:, 0:kk], np.diag(s[0:kk]), vT[:, 0:kk]
        print("\n up :\n", up, "\nsp :\n", sp, "\nvp :\n", vp.T)

        print(np.shape(up))
        print(np.shape(sp))
        print(np.shape(vp.T))
        # Ap = up*up*vp
        Ap = (up.dot(sp)).dot(vp.T)

        print("AP:::\n", Ap)
        print(np.shape(Ap))
        
        G = []
        z = []
        y = []
        avg_list = []
        nn = len(Ap[0])
        
        print("Ap after zigmoid numbers ::\n", Ap)
        for w in range(len(Ap)):
            for q in range(len(Ap[w])):
                if Ap[w][q] < 0 or Ap[w][q] > 1:
                    # print(dist_out[w][q])
                    xz = 1 / (1 + math.exp(-(Ap[w][q])))
            
                    Ap[w][q] = xz
        print("Ap after that :: ", Ap)
        dist_out = pairwise_distances(Ap, metric="cosine")
        #print("dist_cout ::",dist_out)
        #print("len(VT_reduced_to_k[0]) : ",len(VT_reduced_to_k[0]))
        #print("len(VT_reduced_to_k) : ",len(VT_reduced_to_k))
        '''for j in range(len(Ap[0]) - 1):
            
            z.clear()
            
            # print(V[0][j])
            for kk in range(len(Ap)):
                z.append(Ap[kk][j])
            
            # print("\n z' vlaues :",z,"\n")
            #print("j = ",j)
            # zz = np.zeros(zz.shape)
            zz = np.array([np.array(a) for a in z])
            #print("zz : ",zz)
            
            
            for l in range(len(Ap[0]) - (j + 1)):
                for m in range(len(Ap)):
                    y.append(Ap[m][l + j + 1])
                #print("L :",l)
                yy = np.array([np.array(a) for a in y])
                # print("y: ",y,"\n")
                #print("yy : \n",yy,"\n")
                w = 0
                Q = 0
                M = 0
               
                for b in range(len(Ap)):
                    w = ((zz[b] * yy[b])) + w
                #print("w: ((zz[b] * yy[b])) + w ", w)

                # print("V[0][",j,"]","V[1][",j,"]", cos_sim)
                
                for a in range(len(Ap)):
                    Q = (zz[a] ** 2) + Q
                    M = (yy[a] ** 2) + M
                #print("Q,M :",Q,M)
                
                yy = np.zeros(yy.shape)
                y.clear()
                QQ = math.sqrt(Q)
                
                MM = math.sqrt(M)
                #print("QQ,MM :",QQ,MM)
                QM = QQ * MM
                #print("Type(QM): ",type(QM))
                #print("QM = QQ * MM : ",QM)
                
                cos_sim = w / QM
                #print("w :",w)
                #print("cos_sim = w / QM : " ,cos_sim)
                # yy = np.zeros(yy.shape)
                # zz = np.zeros(zz.shape)
                G.append(cos_sim)'''
        sum_G = 0
        count1=0

        '''for w in range(len(dist_out)):
            for q in range(len(dist_out[w])):
                if dist_out[w][q] < 0 or dist_out[w][q] > 1:
                    #print(dist_out[w][q])
                    xz = 1 / (1 + math.exp(-(dist_out[w][q])))
            
                    dist_out[w][q] = xz
        print("dist after that :: ",dist_out)'''
        for ww in range(len(Ap)):
            for qq in range(len(Ap[ww])):
                sum_G = sum_G+Ap[ww][qq]
                count1 = count1+1
        print("sumG: ",sum_G)
        print("len(dist_out)",len(Ap))
        #print("lenG :",len(G))
        sum_avg_G = sum_G / (count1)
        print("sum_avg_G : ", sum_avg_G)
        #if sum_avg_G<0:
           #frac, whole = math.modf(sum_avg_G)
           #PLM = 1 + frac
        #else:
        PLM =1-sum_avg_G  ## PLM includes the probability of being long method
        print("PLM = 1-sum_avg_G :", PLM)
        
            #if (PLM < 0.99): #if name11 not in firstListOfFilel:
               
        firstListOfFilel.append(name11)
        proba_list.append(PLM)
        #elif PLM>=1:
            #listoffile.append(name)
           # count_name = count_name+1
        
#print("list OF greater than 1: ",listoffile)
#print("count_name :: ",count_name)

# get the median of proba_list
# print("the median of proba'z : ",statistics.median(proba_list))
## import values in CSV

## making csv file of prob and y_act and method or file names
#print("list of proba'z : ", proba_list)  ## list includes all probabilities of all methods
# print("y_act:",y_act)
# proba_list = [0.68, 0.80, 0.78]

## make the csv file of actualCSV : method_name and y_act

thresh = statistics.median(proba_list)
print("the threshold of prob'z : ", thresh)
print("len of all probabilities",len(proba_list))
######### Input retrieved method in a list

with open('outfile_prob_of_method.csv', 'w') as f:
    w = csv.writer(f)
    for row in zip(firstListOfFilel):
        w.writerow(row)
#######

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


print("y_actt: ", y_actt)
print("Len(y_actt) :: ", len(y_actt))

y_act_count = 0
for m in range(len(y_actt)):
    if y_actt[m] == '1':
        y_act_count = y_act_count + 1
print("y_act_count :", y_act_count)
print("proba_list :", proba_list)
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
    
    wr.writerows(zip(y_actt, proba_list, firstListOfFilel))

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
    for jj in range(len(proba_list)):
        if proba_list[jj] > thresh:
            thcount = thcount + 1
            List_Of_smells.append(proba_list[jj])
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