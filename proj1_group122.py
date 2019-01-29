import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import time
import math
import matplotlib.pyplot as plt

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

def pre_process(data_set, flag):
    #modularise data parts. 
    word_list=[]
    populatiry_score=[]
    children=[]
    controversiality=[]
    is_root=[]
    for point in data_set:
        point["text"]= point["text"].lower().split()            #changing text into a list of words 
        word_list.append(point["text"])                         #making a list of all the words in our data set
        is_root.append(1 if point['is_root'] else 0 )           #change is_root feature to binary feature
        children.append(point["children"])                      #keep children as quantitavice feature
        controversiality.append(point["controversiality"])      #keep feature as binary featire
        populatiry_score.append(point["popularity_score"])      #keep popularity score as number

    if (flag == 1):
        word_list=list(np.concatenate(word_list))                   #change 2D list into one list
        word_info=Counter(word_list)                                #use counter function to get measure of each word in dataset
        pre_process.most_common_words=word_info.most_common(160)                #get 160 most common words in the data set we are working with
    
    matrix=[]   

    for point in data_set:
        matrix.append([point["text"].count(w[0]) for w in pre_process.most_common_words])   #filling the count matrix row by row
    
    mat=np.array(matrix)

    for x in [is_root,controversiality,children]:
        mat=(np.hstack((mat, np.matrix(x).transpose())))        #adding the other features to our deisgn matrix. 
    
    mat = np.hstack((np.ones((mat.shape[0],1)), mat))           #offset term
    
    return (mat, np.matrix(populatiry_score).transpose())       # the return a tuple of a matrix (160 # of words, is_root, Controversiality, children) and the response variable

def preprocess2(data_set, flag):
    #modularise data parts. 
    word_list=[]
    populatiry_score=[]
    children=[]
    controversiality=[]
    is_root=[]

    feature1=[]
    feature2=[]
    feature3=[]

    for point in data_set:
        point["text"]= point["text"].lower().split()            #changing text into a list of words 
        word_list.append(point["text"])                         #making a list of all the words in our data set
        is_root.append(1 if point['is_root'] else 0 )           #change is_root feature to binary feature
        children.append(point["children"])                      #keep children as quantitavice feature
        controversiality.append(point["controversiality"])      #keep feature as binary featire
        feature1.append(len(point["text"])*point["children"])   #seems we should take into account the size of each text

        root=1 if point['is_root'] else 0 
        feature2.append(root*point["controversiality"])         #controversial root comments would tend to have more popularity
        feature3.append(root*point["children"])                 #number of children to root comment tells you how popular it is


        populatiry_score.append(point["popularity_score"])      #keep popularity score as number

    if (flag == 1):
        word_list=list(np.concatenate(word_list))                   #change 2D list into one list
        word_info=Counter(word_list)                                #use counter function to get measure of each word in dataset
        pre_process.most_common_words=word_info.most_common(160)                #get 160 most common words in the data set we are working with
    
    matrix=[]   

    for point in data_set:
        matrix.append([point["text"].count(w[0]) for w in pre_process.most_common_words])   #filling the count matrix row by row
    
    mat=np.array(matrix)

    for x in [is_root,controversiality,children,feature1,feature2,feature3]:
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

def main():
    
    # load data
    filename = "proj1_data.json"
    training_data, validation_data, testing_data = load_data(filename)
    
    pre_process.most_common_words = list()
    # pre-process data
    train = pre_process(training_data, 1)
    X_train = train[0]
    y_train = train[1]
    
    valid = pre_process(validation_data, 0)
    X_valid = valid[0]
    y_valid = valid[1]
    
    test = pre_process(testing_data, 0)
    X_test = test[0]
    y_test = test[1]
    
    # evaluate performance & regress
    print('----------------------- Evaluation.1 ---------------------------')
    X_train_0 = np.array(X_train)[:,160:]
    X_valid_0 = np.array(X_valid)[:,160:]
    
    start = time.time_ns()/ (10 ** 9)
    w_cf = reg_closed_form(X_train_0, y_train)   
    end = time.time_ns() / (10 ** 9)
    cf_exec_time = end-start
    print('time in sec for wcf(3 features) ', cf_exec_time)
    
    y_cf_valid_pred = np.matmul(X_valid_0, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error_cf(3 features): ", math.sqrt(error_cf))
    
    for n in range(50,250,50):
        start = time.time_ns() / (10 ** 9)
        w_gd = reg_grad_desc(X_train_0, y_train, n)
        end = time.time_ns() / (10 ** 9)
        print('time in sec for w_gf(3 features) %f for beta %d'%(end-start, n))
    
        y_gd_valid_pred = np.matmul(X_valid_0, w_gd)
        #error_gd = np.linalg.norm(y_gd_valid_pred - y_valid)
        error_gd = mean_squared_error(y_valid, y_gd_valid_pred)
        print("RMSE error_gd %f(3 features) for beta %d"%(math.sqrt(error_gd), n))
    print('')
    
    print('----------------------- Evaluation.2 ---------------------------')
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
    print('exec time in sec for wcf(60+ features) ', cf_exec_time)
    
    y_cf_valid_pred = np.matmul(X_valid_60, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error_cf(60+ features)  for validation set: ", math.sqrt(error_cf))
    print('')
    
    y_cf_train_pred = np.matmul(X_train_60, w_cf)
    error_cf = mean_squared_error(y_train, y_cf_train_pred)
    print("RMSE error_cf(60+ features) for train set: ", math.sqrt(error_cf))
    print('')

    print('--------------------------------------------------')
    start = time.time_ns()/ (10 ** 9)
    w_cf = reg_closed_form(X_train, y_train)   
    end = time.time_ns() / (10 ** 9)
    cf_exec_time = end-start
    print('exec time in sec for wcf(160+ features): ', cf_exec_time)
    
    y_cf_valid_pred = np.matmul(X_valid, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    print("RMSE error_cf(160+ features) for validation set: ", math.sqrt(error_cf))
    print('')
    
    y_cf_train_pred = np.matmul(np.array(X_train), w_cf)
    error_cf = mean_squared_error(y_train, y_cf_train_pred)
    print("RMSE error_cf(160+ train) for train set: ", math.sqrt(error_cf))
    print('')
    print('----------------------- Evaluation.3 ---------------------------')
    print('')
    print('')
    print('----------------------- Evaluation.4 ---------------------------')
    print('')
    y_cf_test_pred = np.matmul(X_test, w_cf)
    error_cf = mean_squared_error(y_test, y_cf_test_pred)
    print("RMSE error_cf(160+ features) for test set: ", math.sqrt(error_cf))
    print('')
    # Visualising the Test set results
""" plt.plot(X_valid_0[:,0], y_valid, color = 'red')
    plt.plot(X_valid_0[:,0], y_cf_valid_pred, color = 'blue')
    plt.title('cf, 3 features model')
    plt.xlabel('is_root')
    plt.ylabel('popularity')
    plt.show()
"""
if __name__ == '__main__':
    main()
