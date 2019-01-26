import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
import numpy as np
import sys
from sklearn.metrics import mean_squared_error

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

def reg_closed_form(X_train, y_train):
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.transpose(), X_train)), X_train.transpose()), y_train)
    return w

def reg_grad_desc(X_train, y_train):
    i = 0
    beta = 100
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
    
    # regress
    w_cf = reg_closed_form(X_train, y_train)   
    w_gd = reg_grad_desc(X_train, y_train)
    
    # evaluate performance
    y_cf_valid_pred = np.matmul(X_valid, w_cf)
    #error_cf = np.linalg.norm(y_cf_valid_pred - y_valid)
    error_cf = mean_squared_error(y_valid, y_cf_valid_pred)
    y_gd_valid_pred = np.matmul(X_valid, w_gd)
    #error_gd = np.linalg.norm(y_gd_valid_pred - y_valid)
    error_gd = mean_squared_error(y_valid, y_gd_valid_pred)
    
    print(error_cf)
    print(error_gd)

if __name__ == '__main__':
    main()
