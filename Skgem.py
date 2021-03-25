# Packages:
import pandas as pd
import numpy as np
import json

# Reading Data files:
with open('Data/tallinn/tall_locs.json') as jsonfile:
    locs = json.load(jsonfile)
    
with open('Data/tallinn/tall_flickr_annot_raw.json') as jsonfile:
    annot = json.load(jsonfile)
    
with open('Data/tallinn/tall_poly38_named.json') as jsonfile:
    poly = json.load(jsonfile)
    
with open('Data/tallinn/tall_tags.json') as jsonfile:
    tags = json.load(jsonfile)
  
    
with open('Data/tallinn/tall_userd.json') as jsonfile:
    userd = json.load(jsonfile)
    
    
with open('Data/tallinn/tall_woeid.json') as jsonfile:
    woeid = json.load(jsonfile)
   
# Node2Vec
## Graph
### Nodes:

  # - User
  # - POI
  # - Images
  # - Time
  # - Region (Where on earth)
  # - Tags (Words)
  # - View
  # - Category
  
### Edges:

  # - User-POI
  # - POI-Category
  # - POI-Image
  # - POI-POI
  # - Image-Tag
  # - Image-Time(hour)
  # - POI-VP(View)
  # - Image-Woeid(Region)
  # - Woeid-Woeid

  
# Graph Creation

## We will use the annotation file to create the graph:

import networkx as nx
# Create a graph
G = nx.DiGraph()

test = {}
for user in annot:    
    train = annot[user][0:int(len(annot[user])*0.8)]
    test[user] = annot[user][int(len(annot[user])*0.8):len(annot[user])]
    for tour in range(len(train)):
        for poi in range(len(train[tour])):
    
            
            try:
                # User-POI
                G.add_edge('user_'+str(user),'POI_' + str(train[tour][poi]['poi3'][0]))
    
                # POI-Cat
                cats = str(train[tour][poi]['poi3'][1]['tag']).split('/')
                for cat in cats:
                    G.add_edge('POI_'+str(train[tour][poi]['poi3'][0]),'Cat_'+str(cat).replace(' ',''))

                for pts in range(len(train[tour][poi]['pts'])):
                    # POI-Image
                    G.add_edge('POI_'+str(train[tour][poi]['poi3'][0]),'imgid_' + str(train[tour][poi]['pts'][pts]['imgid']))
                    # Image-Time(hour)
                    G.add_edge('imgid_' + str(train[tour][poi]['pts'][pts]['imgid']),pd.to_datetime(train[tour][poi]['pts'][pts]['time']).hour)
                
                # POI-VP   
                try:
                    G.add_edge('POI_'+str(train[tour][poi]['poi3'][0]),'VP_'+str(train[tour][poi]['roi2'][0]))
                except:
                    pass
                
                try:
                    if poi > 0:
                        # POI-POI
                        G.add_edge('POI_'+str(train[tour][poi-1]['poi3'][0]),'POI_'+str(train[tour][poi]['poi3'][0]))
                except:
                    G.add_edge('VP_'+str(train[tour][poi-1]['roi2'][0]),'POI_'+str(train[tour][poi]['poi3'][0]))

            except:
                try:
                    G.add_edge('user_'+str(user),'VP_'+str(train[tour][poi]['roi2'][0]))
                except:
                    pass
                
        
        for tour in  range(len(test[user])):
            for poi in range(len(test[user][tour])):
                try:
                    G.add_node('POI_'+str(test[user][tour][poi]['poi3'][0]))
                except:
                    pass
            
            
for img in tags:
    # Image-Tag
    for tag in tags[img]:
        G.add_edge('imgid_' + str(img),'Tag_'+str(tag))
                           
for img in locs:
    # Image-Woeid
    G.add_edge('imgid_' + str(img),'woeid_'+str(locs[img]['woeid']))
    for name in locs[img]['names']:
        G.add_edge(locs[img]['woeid'], 'Region_'+str(name))

    for woeid in locs[img]['woeids']:
        # Woeid-Woeid
        G.add_edge(locs[img]['woeid'],'woeid_'+str(woeid))
  
# Node2Vec Model definition:
  
from node2vec import Node2Vec
# Generate walks
node2vec = Node2Vec(G, dimensions=80, walk_length=4, num_walks=300)

# Learn embeddings 
model = node2vec.fit(window=4, min_count=2)

##  Saving the model:

# Save embeddings for later use:
model.wv.save_word2vec_format('ModelVectors')

# Save model for later use
model.save('EmbeddingModel')

# Create users evaluation visited POI in test:
# Each user: [poi,cat,time]
Visited_test = {}
for user in test:
    visited_test = []
    for tour in range(len(test[user])):
        for poi in range(len(test[user][tour])):
            try: 
                visited_test.append(['POI_'+str(test[user][tour][poi]['poi3'][0]),'Cat_'+str(test[user][tour][poi]['poi3'][1]['tag']),[pd.to_datetime(test[user][tour][poi]['pts'][x]['time'], unit='s') for x in range(len(test[user][tour][poi]['pts']))]])
            except:
                pass
            
            
#     print(visited_test)
    Visited_test[user] = visited_test
    
    
    
# Create users evaluation visited POI in train:
# Each user: [poi,cat,time]
Visited_train = {}
for user in annot:  
    train = annot[user][0:int(len(annot[user])*0.8)]
    visited_train = []
    for tour in range(len(train)):
        for poi in range(len(train[tour])):
            try: 
                visited_train.append(['POI_'+str(train[tour][poi]['poi3'][0]),'Cat_'+str(train[tour][poi]['poi3'][1]['tag']),[pd.to_datetime(train[tour][poi]['pts'][x]['time'], unit='s') for x in range(len(train[tour][poi]['pts']))]])
            except:
                pass
            
    if len(train)==0:
        try:
            visited_train.append(['POI_'+str(annot[user][0][0]['poi3'][0]),'Cat_'+str(annot[user][0][0]['poi3'][1]['tag']),[pd.to_datetime(annot[user][0][0]['pts'][x]['time'], unit='s') for x in range(len(annot[user][0][0]['pts']))]])
        except:
            pass
            
            
    Visited_train[user] = visited_train
    
    
def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result
  
  
mean_p_k_1 = np.mean([RecSys[user][0]['1'] for user in RecSys])
mean_p_k_5 = np.mean([RecSys[user][0]['5'] for user in RecSys])
mean_p_k_10 = np.mean([RecSys[user][0]['10'] for user in RecSys])
mean_p_k_15 = np.mean([RecSys[user][0]['15'] for user in RecSys])
mean_p_k_20 = np.mean([RecSys[user][0]['20'] for user in RecSys])


mean_r_k_1 = np.mean([RecSys[user][1]['1'] for user in RecSys])
mean_r_k_5 = np.mean([RecSys[user][1]['5'] for user in RecSys])
mean_r_k_10 = np.mean([RecSys[user][1]['10'] for user in RecSys])
mean_r_k_15 = np.mean([RecSys[user][1]['15'] for user in RecSys])
mean_r_k_20 = np.mean([RecSys[user][1]['20'] for user in RecSys])


print('Mean Precision@1: ',mean_p_k_1)
print('Mean Precision@5: ',mean_p_k_5)
print('Mean Precision@10: ',mean_p_k_10)
print('Mean Precision@15: ',mean_p_k_15)
print('Mean Precision@20: ',mean_p_k_20)
print()
print('Mean Recall@1: ',mean_r_k_1)
print('Mean Recall@5: ',mean_r_k_5)
print('Mean Recall@10: ',mean_r_k_10)
print('Mean Recall@15: ',mean_r_k_15)
print('Mean Recall@20: ',mean_r_k_20)
  
  
# Primitive function: find the most populaire POIs.
def primitive(checkins,k):
    # we will consider the most visited as the most populaire, k ipoi.
    result = checkins[1].value_counts()[:k].index.tolist()
    for poi in range(len(result)):
        result[poi] = 'POI_'+str(result[poi])
    return result
    
    
# Serendibity = mean(intersection(predicted, actual-primitive(checkins,k))/k)
def serendibity(actual,predicted, k, checkins):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    mv_set= set(primitive(checkins,k))
    result = len(pred_set & (act_set - mv_set)) / float(k)
    return result

  
evl = [1 , 5, 10, 15, 20]
RecSys = {}
for user in Visited_train:
    
    pk_dict = {}
    rk_dict = {}
    sk_dict = {}

    visited_train = [Visited_train[user][x][0] for x in range(len(Visited_train[user]))]
    visited_test = [Visited_test[user][x][0] for x in range(len(Visited_test[user]))]
    
    predictions = model.predict_output_word(visited_train, topn=20)
    
    pois = []
    try:
        for p in predictions:
            if 'POI_' in str(p[0]):
                pois.append(p[0])

        for k in evl:

            sk_dict[str(k)] = serendibity(visited_test, pois,k,checkins)
    except:
        for k in evl:
           
            sk_dict[str(k)] = 0.0

        pass
    

    RecSys[user] = [sk_dict]
    
    
mean_s_k_1 = np.mean([RecSys[user][0]['1'] for user in RecSys])
mean_s_k_5 = np.mean([RecSys[user][0]['5'] for user in RecSys])
mean_s_k_10 = np.mean([RecSys[user][0]['10'] for user in RecSys])
mean_s_k_15 = np.mean([RecSys[user][0]['15'] for user in RecSys])
mean_s_k_20 = np.mean([RecSys[user][0]['20'] for user in RecSys])

print('Mean Serendibity@1: ',mean_s_k_1)
print('Mean Serendibity@5: ',mean_s_k_5)
print('Mean Serendibity@10: ',mean_s_k_10)
print('Mean Serendibity@15: ',mean_s_k_15)
print('Mean Serendibity@20: ',mean_s_k_20)
print()


import math
# # freq function: find the probabilty of a POI in the train set.
def freq(Visited_train,poi):
    # nom d'occ d'un item /nombre de sequences in train set
    observ = 0
    for user in Visited_train:
        visited_train = [Visited_train[user][x][0] for x in range(len(Visited_train[user]))]
        if poi in visited_train:
            observ = observ + 1
            
    result = float(observ)/ float(len(Visited_train))
   
    return result
    
    
# Novelty = mean(log fre(poi))
def novelty(predicted, k, Visited_train):
    seq = []
    
    for poi in range(len(predicted[:k])):
#             print('Freq: ',float(freq(Visited_train,predicted[poi])))
        if float(freq(Visited_train,predicted[poi])) == float(0.0):
            seq.append(0.0)
        else:
            seq.append(math.log(float(freq(Visited_train,predicted[poi])),2))

            
    if not seq:
        seq.append(0.0)
            
#     print('Seq: ', seq)
    result = np.mean(seq)  
#     if math.isnan(result):
#         print('nan')
    return result


# Make Recommendation:
# Extract the most similar POI vectors to the users vectors 
evl = [1 , 5, 10, 15, 20]
RecSys = {}
for user in Visited_train:
    
    nk_dict = {}

    visited_train = [Visited_train[user][x][0] for x in range(len(Visited_train[user]))]
    visited_test = [Visited_test[user][x][0] for x in range(len(Visited_test[user]))]
    
    predictions = model.predict_output_word(visited_train, topn=20)
    
    pois = []
    try:
        for p in predictions:
            if 'POI_' in str(p[0]):
                pois.append(p[0])

        for k in evl:
#             print(novelty(pois,k,Visited_train))
            nk_dict[str(k)] = novelty(pois,k,Visited_train)
    except:
        for k in evl:
            nk_dict[str(k)] = 0.0
        pass
    

    RecSys[user] = [nk_dict]
    
mean_n_k_1 = np.mean([RecSys[user][0]['1'] for user in RecSys])
mean_n_k_5 = np.mean([RecSys[user][0]['5'] for user in RecSys])
mean_n_k_10 = np.mean([RecSys[user][0]['10'] for user in RecSys])
mean_n_k_15 = np.mean([RecSys[user][0]['15'] for user in RecSys])
mean_n_k_20 = np.mean([RecSys[user][0]['20'] for user in RecSys])


print('Mean Novelty@1: ',mean_n_k_1)
print('Mean Novelty@5: ',mean_n_k_5)
print('Mean Novelty@10: ',mean_n_k_10)
print('Mean Novelty@15: ',mean_n_k_15)
print('Mean Novelty@20: ',mean_n_k_20)
print()



# (2*pairs-(predicted,actual)  +  pairsu(predicted,actual))/ 2*pairs(predicted)

# Pairs-:
import itertools
from iteration_utilities import unique_everseen

def pairsminus(actual,predicted,k):
    # Remove duplicates in the actual and predicted lists:
    actual = list(dict.fromkeys(actual))
    predicted = list(dict.fromkeys(predicted))
   
    
    all_act_pairs = []
    if len(actual)>=k:
        for pair in itertools.combinations(actual[:k],2):
            all_act_pairs.append(pair)
    else: 
        if len(actual)==1:
            all_act_pairs.append((actual[0],actual[0]))

        else:
            for pair in itertools.combinations(actual,2):
                all_act_pairs.append(pair)

    all_pred_pairs = []
    if len(predicted)>=k:
        for pair in itertools.combinations(predicted[:k],2):
            all_pred_pairs.append(pair)
    else: 
        if len(actual)==1:
            all_pred_pairs.append((actual[0],predicted[0]))

        else:
            for pair in itertools.combinations(predicted,2):
                all_pred_pairs.append(pair)
                   
    
    
    result = 0  
    for el in all_act_pairs:
        if (el[1],el[0]) in all_pred_pairs:
            result = result + 1 
            
    return float(result)
        

# Pairsu:
from iteration_utilities import unique_everseen
def pairsu(actual,predicted,k):
    
    actual = list(dict.fromkeys(actual))
    predicted = list(dict.fromkeys(predicted))

    
    all_act_pairs = []
    for pair in itertools.combinations(actual[:k],2):
        all_act_pairs.append(pair)
        
    all_pred_pairs = []
    for pair in itertools.combinations(predicted[:k],2):
        all_pred_pairs.append(pair)
        
    # The order is not important:
    all_act_pairs = list(unique_everseen(all_act_pairs, key=set))
    all_pred_pairs = list(unique_everseen(all_pred_pairs, key=set))

    
    result = 0  
    for el in all_act_pairs:
        if el in all_pred_pairs:
            result = result + 1 
            
    return float(result)

# Pairs:
def pairs(actual,k):
    all_act_pairs = []
    if len(actual)>=k:
        for pair in itertools.combinations(actual[:k],2):
            all_act_pairs.append(pair)
    else: 
        if len(actual)==1:
            all_act_pairs.append((actual[0],actual[0]))
        else:
            for pair in itertools.combinations(actual,2):
                all_act_pairs.append(pair)


    all_act_pairs = list(unique_everseen(all_act_pairs, key=set))
    return float(len(all_act_pairs))

# nDPM = (2*pairs-(predicted,actual)  +  pairsu(predicted,actual))/ 2*pairs(predicted)

def nDPM(actual,predicted,k):
    result = (2 * pairsminus(actual,predicted,k) + pairsu(actual,predicted,k))/ (2*pairs(actual,k))
    return result
# Make Recommendation:
# Extract the most similar POI vectors to the users vectors 
evl = [5, 10, 15, 20]
RecSys = {}
for user in Visited_train:
    
    nDPMk_dict = {}
    visited_train = [Visited_train[user][x][0] for x in range(len(Visited_train[user]))]
    visited_test = [Visited_test[user][x][0] for x in range(len(Visited_test[user]))]
    predictions = model.predict_output_word(visited_train, topn=20)
    
    pois = []
    try:
        for p in predictions:
            if 'POI_' in str(p[0]):
                pois.append(p[0])

        for k in evl:
            nDPMk_dict[str(k)] = nDPM(visited_test,pois,k)
    except:
        for k in evl:
            nDPMk_dict[str(k)] = 0.0
        pass
    
    RecSys[user] = [nDPMk_dict]
    

mean_nDPM_k_5 = np.mean([RecSys[user][0]['5'] for user in RecSys])
mean_nDPM_k_10 = np.mean([RecSys[user][0]['10'] for user in RecSys])
mean_nDPM_k_15 = np.mean([RecSys[user][0]['15'] for user in RecSys])
mean_nDPM_k_20 = np.mean([RecSys[user][0]['20'] for user in RecSys])

print('Mean nDPM@5: ',mean_nDPM_k_5)
print('Mean nDPM@10: ',mean_nDPM_k_10)
print('Mean nDPM@15: ',mean_nDPM_k_15)
print('Mean nDPM@20: ',mean_nDPM_k_20)
print()



# Make Recommendation:
# Extract the most similar POI vectors to the users vectors 

import scipy.stats as stats

evl = [5, 10, 15, 20]
RecSys = {}
for user in Visited_train:
    
    kendallk_dict = {}
    visited_train = [Visited_train[user][x][0] for x in range(len(Visited_train[user]))]
    visited_test = [Visited_test[user][x][0] for x in range(len(Visited_test[user]))]
    predictions = model.predict_output_word(visited_train, topn=100)
    
    pois = []
    
    try:
        visited_test = list(dict.fromkeys(visited_test))
        predictions = list(dict.fromkeys(predictions))
        if predictions and visited_test:
            for p in predictions:
                if 'POI_' in str(p[0]):
                    pois.append(p[0])

            for k in evl:   
#                 print(len(visited_test[:k]) , len(pois[:k]) )
                if len(visited_test[:k]) == 1 or len(pois[:k]) == 1:
#                     print("1111111111111111111111111111111111111111111")
                    if visited_test[0] == pois[0]:
                        kendallk_dict[str(k)] = 1.0
                    else:
                        kendallk_dict[str(k)] = -1.0

#                     print(kendallk_dict[str(k)])

                else:

                    if len(visited_test[:k]) == len(pois[:k]):
                        kendallk_dict[str(k)], p_value = stats.kendalltau(visited_test[:k],pois[:k])
#                         print('================')
#                         print(kendallk_dict[str(k)])

                    else: 
                        if len(visited_test[:k]) <= len(pois[:k]):
                            kendallk_dict[str(k)], p_value = stats.kendalltau(visited_test[:k],pois[:len(visited_test)])
#                             print('<<<<<<<<<<<<<<<<<<<<<<<<<<', visited_test[:k],pois[:len(visited_test)])
#                             print(kendallk_dict[str(k)])

                        else: 
                            kendallk_dict[str(k)], p_value = stats.kendalltau(visited_test[:len(pois)],pois[:k])
#                             print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#                             print(kendallk_dict[str(k)])

  
        else:
            for k in evl:
                kendallk_dict[str(k)] = 0.0
    except:
        for k in evl:
            kendallk_dict[str(k)] = 0.0
        pass

    RecSys[user] = [kendallk_dict]
    

mean_kendall_k_5 = np.mean([RecSys[user][0]['5'] for user in RecSys])
mean_kendall_k_10 = np.mean([RecSys[user][0]['10'] for user in RecSys])
mean_kendall_k_15 = np.mean([RecSys[user][0]['15'] for user in RecSys])
mean_kendall_k_20 = np.mean([RecSys[user][0]['20'] for user in RecSys])

print('Mean KendallCoef@5: ',mean_kendall_k_5)
print('Mean KendallCoef@10: ',mean_kendall_k_10)
print('Mean KendallCoef@15: ',mean_kendall_k_15)
print('Mean KendallCoef@20: ',mean_kendall_k_20)
print()











  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
