import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from collections import Counter
import numpy as np
import sys

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

np.set_printoptions(threshold=sys.maxsize)


training_data= data[0:9999]
validation_data=data[10000:19999]
testing_data=data[11000:11999]


def preprocess(data_set):
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



    word_list=list(np.concatenate(word_list))                   #change 2D list into one list
    word_info=Counter(word_list)                                #use counter function to get measure of each word in dataset
    most_common_words=word_info.most_common(160)                #get 160 most common words in the data set we are working with

    matrix=[]
    

    for point in data_set:
        matrix.append([point["text"].count(w[0]) for w in most_common_words])   #filling the count matrix row by row
    
    mat=np.array(matrix)

    print(np.matrix(children).shape)
    for x in [is_root,controversiality,children]:
        mat=(np.hstack((mat, np.matrix(x).transpose())))        #adding the other features to our deisgn matrix. 
    
    return (mat, np.matrix(populatiry_score).transpose())       # the return a tuple of a matrix (160 # of words, is_root, Controversiality, children) and the response variable


