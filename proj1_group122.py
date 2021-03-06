import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def commonwords(training_data):
    word_list=[]
    for point in training_data:
        word_list.append(point["text"].lower().split())
    #making a list of all the words in our data set
    word_list=list(np.concatenate(word_list))                   #change 2D list into one list
    word_info=Counter(word_list)                                #use counter function to get measure of each word in dataset
    most_common_words = word_info.most_common(160)                #get 160 most com
    with open('most.txt', 'w') as f:
        for item in most_common_words:
            f.write(str(item[0]))
            f.write('\n')
    
    f= open('most.txt', 'r')    
    lines=f.read().split('\n')
    f.close()    
    del lines[-1]
    return lines
    
def preprocess2(training_data, validation_data, testing_data):
    #new features
    feature1=[]
    feature2=[]
    feature3=[]
    text_list=[]
    for point in training_data:
        text_list.append(point["text"])
        feature1.append(len(point["text"])*point["children"])   #seems we should take into account the size of each text
        root=1 if point['is_root'] else 0
        feature2.append(root*point["controversiality"])         #controversial root comments would tend to have more popularity
        feature3.append(root*point["children"])                 #number of children to root comment tells you how popular it is

    vect = TfidfVectorizer(max_features=160, max_df=4500, min_df=200)
    dtm = vect.fit_transform(text_list)
    msg0 = pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())
    msg0=msg0.values #numpy array
    msg_0=(msg0,feature1,feature2,feature3)

    #new features
    feature1=[]
    feature2=[]
    feature3=[]
    text_list=[]
    for point in validation_data:
        text_list.append(point["text"])
        feature1.append(len(point["text"])*point["children"])   #seems we should take into account the size of each text
        root=1 if point['is_root'] else 0
        feature2.append(root*point["controversiality"])         #controversial root comments would tend to have more popularity
        feature3.append(root*point["children"])                 #number of children to root comment tells you how popular it is
    validdtm = vect.transform(text_list)
    msg1 = pd.DataFrame(validdtm.toarray(), columns=vect.get_feature_names())
    msg1=msg1.values #numpy array
    msg_1=(msg1,feature1,feature2,feature3)

    feature1=[]
    feature2=[]
    feature3=[]
    text_list=[]
    for point in testing_data:
        text_list.append(point["text"])
        feature1.append(len(point["text"])*point["children"])   #seems we should take into account the size of each text
        root=1 if point['is_root'] else 0
        feature2.append(root*point["controversiality"])         #controversial root comments would tend to have more popularity
        feature3.append(root*point["children"])                 #number of children to root comment tells you how popular it is
    testdtm = vect.transform(text_list)
    msg2 = pd.DataFrame(testdtm.toarray(), columns=vect.get_feature_names())
    msg2=msg2.values #numpy array
    msg_2=(msg2,feature1,feature2,feature3)
    
    return msg_0, msg_1, msg_2

def load_data(filename):
    with open(filename) as fp:
        data = json.load(fp)

    # Now the data is loaded.
    # It a list of data points, where each datapoint is a dictionary with the following attributes:
    # popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
    # children : the number of replies to this comment (type: int)
    # text : the text of this comment (type: string)
    # controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
    # is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

    training_data= data[0:10000]
    validation_data=data[10000:11000]
    testing_data=data[11000:12000]
    
    return training_data, validation_data, testing_data

def pre_process(data_set, lines):
    #modularise data parts. 
    populatiry_score=[]
    children=[]
    controversiality=[]
    is_root=[]
    for point in data_set:
        point["text"]= point["text"].lower().split()            #changing text into a list of words                          #making a list of all the words in our data set
        is_root.append(1 if point['is_root'] else 0 )           #change is_root feature to binary feature
        children.append(point["children"])                      #keep children as quantitavice feature
        controversiality.append(point["controversiality"])      #keep feature as binary featire
        populatiry_score.append(point["popularity_score"])      #keep popularity score as number
   
    matrix=[]   

    for point in data_set:
        matrix.append([point["text"].count(w) for w in lines])   #filling the count matrix row by row
    
    mat=np.array(matrix)

    for x in [is_root,controversiality,children]:
        mat=(np.hstack((mat, np.matrix(x).transpose())))        #adding the other features to our deisgn matrix. 
    
    mat = np.hstack((np.ones((mat.shape[0],1)), mat))           #offset term
    
    return (mat, np.matrix(populatiry_score).transpose())       # the return a tuple of a matrix (160 # of words, is_root, Controversiality, children) and the response variable


def reg_closed_form(X_train, y_train):
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.transpose(), X_train)), X_train.transpose()), y_train)
    return w

def reg_grad_desc(X_train, y_train, beta):
    i = 0
    #beta = 100
    alpha = 1/(1 + beta*i)
    epsilon = 0.2

    w_prev = np.zeros(X_train.shape[1]).reshape(X_train.shape[1], 1)
    w = w_prev - 2*alpha*(np.matmul(np.matmul(X_train.transpose(), X_train), w_prev) - np.matmul(X_train.transpose(), y_train)) 
    norm_diff = np.linalg.norm(w - w_prev)

    while norm_diff > epsilon:
        #print(i)
        #print('norm_diff: ' + str(norm_diff))
        i += 1
        alpha = 1/(1 + beta*i)
        w = w_prev - 2*alpha*(np.matmul(np.matmul(X_train.transpose(), X_train), w_prev) - np.matmul(X_train.transpose(), y_train)) 

        norm_diff = np.linalg.norm(w - w_prev)
        
    return w

def eval1(X_train, X_valid, y_train, y_valid):
    print('_______________________________ Evaluation.1 _____________________________')
    print('')
    X_train_0 = np.array(X_train)[:,160:]
    X_valid_0 = np.array(X_valid)[:,160:]
    
    for n in range(50,250,50):
        start = time.time_ns() / (10 ** 9)
        w_gd = reg_grad_desc(X_train_0, y_train, n)
        end = time.time_ns() / (10 ** 9)
        print('Execution time in sec for W-gradient-form (3 features) %f for beta = %d in learning-rate = (1/(1 + beta*i))'%(end-start, n))
        y_gd_valid_pred = np.matmul(X_valid_0, w_gd)
        #error_gd = np.linalg.norm(y_gd_valid_pred - y_valid)
        error_gd = mean_squared_error(y_valid, y_gd_valid_pred)
        print("RMSE error gradien-form %f(3 features) for beta %d"%(math.sqrt(error_gd), n))
        print('')

    print('---------------------------------------------------------------------------')
    print('')
    start = time.time_ns()/ (10 ** 9)
    w_cf = reg_closed_form(X_train_0, y_train)   
    end = time.time_ns() / (10 ** 9)
    cf_exec_time = end-start
    print('Execution time in sec for W-closed-form (3 features) ', cf_exec_time)
    
    y_cf_valid_pred = np.matmul(X_valid_0, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error closed-form (3 features): ", math.sqrt(error_cf))
    print('')
    
def eval2(X_train, X_valid, y_valid, y_train):
    print('______________________________ Evaluation.2 ________________________________')
    print('')
    temp0 = np.array(X_train)[:, 0:60]
    temp1 = np.array(X_train)[:,160:]
    X_train_60 = np.hstack((temp0,temp1))
    temp0 = np.array(X_valid)[:, 0:60]
    temp1 = np.array(X_valid)[:,160:]
    X_valid_60 = np.hstack((temp0,temp1))
    
    start = time.time_ns()/ (10 ** 9)
    w_cf = reg_closed_form(X_train_60, y_train)   
    end = time.time_ns() / (10 ** 9)
    cf_exec_time = end-start
    print('Execution time in sec for W-closed-form (+60 most-common-words features) ', cf_exec_time)
    print('')
    y_cf_valid_pred = np.matmul(X_valid_60, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error closed-form (+60 most-common-words features)  for validation set: ", math.sqrt(error_cf))
    print('')
    
    y_cf_train_pred = np.matmul(X_train_60, w_cf)
    error_cf = mean_squared_error(y_train, y_cf_train_pred)
    print("RMSE error closed-form (+60 most_common-words features) for train set: ", math.sqrt(error_cf))
    print('')

    print('-----------------------------------------------------------------------')
    print('')
    start = time.time_ns()/ (10 ** 9)
    w_cf = reg_closed_form(X_train, y_train)   
    end = time.time_ns() / (10 ** 9)
    cf_exec_time = end-start
    print('Execution time in sec for W-closed-form(+160 most-common-words features): ', cf_exec_time)
    print('')
    y_cf_valid_pred = np.matmul(X_valid, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error closed-form (+160 most-common-words features) for validation set: ", math.sqrt(error_cf))
    print('')
    
    y_cf_train_pred = np.matmul(np.array(X_train), w_cf)
    error_cf = mean_squared_error(y_train, y_cf_train_pred)
    print("RMSE error closed-form(+160 most-common-words features) for train set: ", math.sqrt(error_cf))
    print('')
 
def eval3(filename, X_train, y_train, X_valid, y_valid, X_test):   
    print('_______________________________ Evaluation.3 __________________________________')
    print('')
    training_data, validation_data, testing_data = load_data(filename)
    msg0, msg1, msg2 = preprocess2(training_data, validation_data, testing_data)
    temp0 = np.array(X_train)[:, 160:]

    temp0=np.hstack((temp0, msg0[0]))
    for feature in msg0[1:]:
        temp0=np.hstack((temp0, np.matrix(feature).transpose()))
    w_cf = reg_closed_form(temp0, y_train)      
    y_cf_train_pred = np.matmul(temp0, w_cf)
    error_cf = mean_squared_error(y_train, y_cf_train_pred)
    print("RMSE error closed-form (160 words-Tfidf + 3 extra features) for Training set: ", math.sqrt(error_cf))
    print('') 

    temp0 = np.array(X_valid)[:, 160:]
    temp0=np.hstack((temp0, msg1[0]))
    for feature in msg1[1:]:
        temp0=np.hstack((temp0, np.matrix(feature).transpose()))
    w_cf = reg_closed_form(temp0, y_valid)      
    y_cf_valid_pred = np.matmul(temp0, w_cf)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error closed-form (160 words-Tfidf + 3 extra features) for Validation set: ", math.sqrt(error_cf))       
    print('')
    return msg2
    
def eval4(msg2, X_test, y_test):
    print('_______________________________ Evaluation.4 ________________________________')
    print('')
    temp0 = np.array(X_test)[:, 160:]
    temp0=np.hstack((temp0, msg2[0]))
    for feature in msg2[1:]:
        temp0=np.hstack((temp0, np.matrix(feature).transpose()))
    w_cf = reg_closed_form(temp0, y_test)      
    y_cf_test_pred = np.matmul(temp0, w_cf)
    print('')
    error_cf = mean_squared_error(y_test, y_cf_test_pred)
    print("RMSE error for closed-form (160 words-Tfidf + 3 extra features) for testing set: ", math.sqrt(error_cf))
    print('')
    dist = np.abs(y_test - y_cf_test_pred)
    index = np.arange(0,1000,1)
    # Visualising the Test set results
    dist = dist.A1
    plt.scatter(index, dist, color = 'blue')
    plt.title('closed-Form, abs Distance vs Index for Test-Data')
    plt.xlabel('index')
    plt.ylabel('abs distance')
    plt.show()
    
def main():
    
    # load data
    filename = "proj1_data.json"
    training_data, validation_data, testing_data = load_data(filename)
    

    
    lines = commonwords(training_data)
    # pre-process data
    train = pre_process(training_data, lines)
    X_train = train[0]
    y_train = train[1]
    
    valid = pre_process(validation_data, lines)
    X_valid = valid[0]
    y_valid = valid[1]
    
    test = pre_process(testing_data, lines)
    X_test = test[0]
    y_test = test[1]
    
    # evaluate performance & regress
    eval1(X_train, X_valid, y_train, y_valid)
    eval2(X_train, X_valid, y_valid, y_train)
    msg2 = eval3(filename, X_train, y_train, X_valid, y_valid, X_test)
    eval4(msg2, X_test, y_test)

if __name__ == '__main__':
    main()
