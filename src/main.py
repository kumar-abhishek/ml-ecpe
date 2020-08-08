# -*- coding: utf-8 -*-

"""https://stackoverflow.com/questions/55956200/integrate-keras-to-sklearn-pipeline
sklearn pipeline 
https://queirozf.com/entries/scikit-learn-pipeline-examples
"""
import sys
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from create_dataset import *
from ecmodel import first_layer_model
from filter_model import create_input_filter_model, logistic_filtering_model
from sklearn.metrics import classification_report

emotion_seeds = list(["happy", "sad", "surprise", "disgust", "anger", "fear", "shame"])

num_docs=225   # reset later to more, maybe 500 or 800
num_epochs=100

def split_train_test(X, y_cause, y_emotion):
  all_indices = list(range(len(X)))
  train_indices, test_indices = train_test_split(all_indices, test_size=0.2)
  print('train_indices: ', train_indices)
  print('test_indices: ', test_indices)
  X_train, X_test = [], []
  y_cause_train, y_cause_test = [], []
  y_emotion_train, y_emotion_test = [], []

  
  for index in train_indices:
    try:
      X_train.append(X[index])
      y_cause_train.append(y_cause[index])
      y_emotion_train.append(y_emotion[index])
    except Exception as e:
      print("run in to exception")
      print(clause_global[index])
      print(e)
      sys.exit(-1)

  for index in test_indices:
    X_test.append(X[index])
    y_cause_test.append(y_cause[index])
    y_emotion_test.append(y_emotion[index])

  X_train, X_test, y_cause_train, y_cause_test, y_emotion_train, y_emotion_test = \
      list(map(np.asarray, [X_train, X_test, y_cause_train, y_cause_test, y_emotion_train, y_emotion_test]))

  return X_train, X_test, y_cause_train, y_cause_test, y_emotion_train, y_emotion_test

def get_emotion_id_from_emotion_seed(cur_emotion):
  for id, emotion in enumerate(emotion_seeds):
    if emotion==cur_emotion:
      return id

path_to_file = "input/data_labeled_emotions.txt"

lines = io.open(path_to_file, encoding='UTF-8').read().strip().split('\n')

document, cause, clause_list, clause_global, cause_label, emotion_label = create_dataset(lines, num_docs)

print('number of documents: ', len(document),'\n')

# print bug; todo: print till len(clauses) if you want to see all documents.
for i in range(len(cause_label)): 
  print('clause: ', clause_global[i])
  print('cause label: ', cause_label[i])
  #print('emotion label: ', emotion_label[i])
  print('clauseid_to_docid label: ', clauseid_to_docid[i])
  print('clause_id: ', i,  '| clause:', clause_global[i])
  print('\n--------\n')


for i in range(len(known_emotion_cause_pair_per_doc_id)):
  print('i: ', i, ' | known_emotion_cause_pair_per_doc_id: ',  known_emotion_cause_pair_per_doc_id[i], '| corresponding cause clause :',  clause_global[known_emotion_cause_pair_per_doc_id[i][1]])

for i in range(len(emotion_label)):
  print('clause_global[i]: ', clause_global[i], ' | emotion_label[i]: ', emotion_label[i])

known_emotion_cause_pair_per_doc_id[:500]

X=clause_global
y_cause=np.asarray(cause_label)
y_emotion=np.asarray(emotion_label)

X, X_test, y_cause, y_cause_test, y_emotion, y_emotion_test = split_train_test(X, y_cause, y_emotion)

print(clause_global[:10])
print( type(clause_global))

print('y_cause: ', y_cause, type(y_cause), len(y_cause))
print('y_emotion: ', y_emotion, len(y_emotion))
print('len(X_test):', len(X_test))

tokenizer_emotion=keras.preprocessing.text.Tokenizer(num_words=7, oov_token="xxxxxxx") # TODO: check num_words 
tokenizer_emotion.fit_on_texts(emotion_seeds)
print(len(tokenizer_emotion.word_index), len(emotion_seeds), tokenizer_emotion.word_index)
tokenized_id_to_emotion_map={}
for emotion, idx in tokenizer_emotion.word_index.items():
  tokenized_id_to_emotion_map[idx] = emotion


def tokenize_y_emotion(ye):
  y_emotion_tokenized=[]
  for emotion in ye:
    if emotion in tokenizer_emotion.word_index:
      y_emotion_tokenized.append(tokenizer_emotion.word_index[emotion])
    else:
      y_emotion_tokenized.append(1)
  y_emotion_tokenized = np.asarray(y_emotion_tokenized)
  return y_emotion_tokenized

y_emotion_tokenized=tokenize_y_emotion(y_emotion)
y_emotion_test_tokenized=tokenize_y_emotion(y_emotion_test)

print(y_emotion_tokenized[:200], len(y_emotion_tokenized))

print('tokenized_id_to_emotion_map: ', tokenized_id_to_emotion_map)
print('tokenizer_emotion.word_index: ', tokenizer_emotion.word_index)

# Fix known_emotion_cause_pair_per_doc_id to use tokenized emotion indices
print('known_emotion_cause_pair_per_doc_id: ', known_emotion_cause_pair_per_doc_id)
fixed_known_emotion_cause_pair_per_doc_id = []
for idx, item in enumerate(known_emotion_cause_pair_per_doc_id):
  emotion, cause = item[0], item[1]
  fixed_known_emotion_cause_pair_per_doc_id.append(known_emotion_cause_pair_per_doc_id[idx])
  print("emotion: ", emotion, "| idx:", idx)
  fixed_known_emotion_cause_pair_per_doc_id[idx][0] = tokenizer_emotion.word_index[emotion] #tokenizer_emotion.transform([[emotion_seeds[emotion]]])[0] 
  #print('emotion_seeds[emotion]: ',emotion_seeds[emotion] ,'idx:', idx, 'emotion: ', emotion, 'cause:', cause, 'new_emotion: ', fixed_known_emotion_cause_pair_per_doc_id[idx][0])

print('fixed_known_emotion_cause_pair_per_doc_id:', fixed_known_emotion_cause_pair_per_doc_id)

known_emotion_cause_pair_per_doc_id[:30]

tokenizer=keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="xxxxxxx")
tokenizer_emotion=keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="xxxxxxx")

tokenizer.fit_on_texts(X)
tokenizer_emotion.fit_on_texts(emotion_seeds)
X_dict=tokenizer.word_index

X_seq=tokenizer.texts_to_sequences(X)
X_test_seq=tokenizer.texts_to_sequences(X_test)
X_padded_seq=pad_sequences(X_seq,padding='post',maxlen=30)
X_test_padded_seq=pad_sequences(X_test_seq,padding='post',maxlen=30)
print(X_padded_seq[:6], X_padded_seq.shape, type(X_padded_seq))
print(X_padded_seq.shape)

print(len(X_padded_seq))
print(len(y_emotion_tokenized))
print(len(y_cause))

#print(len(X_padded_seq), len(X_padded_seq_test), len(y_emotion_tokenized), len(y_emotion_tokenized_test), len(y_cause), len(y_cause_test))
print(X_padded_seq[:6],y_emotion_tokenized[:6], y_cause[:6] )

model = first_layer_model(X_padded_seq, tokenizer_emotion, y_emotion_tokenized, y_cause, num_epochs)

(predicted_emotions, predicted_causes) = model.predict({'clause_input': X_padded_seq})
(predicted_emotions_test, predicted_causes_test) = model.predict({'clause_input': X_test_padded_seq})

print("len(X_test_padded_seq):")
print(len(X_test_padded_seq))
logistic_model_per_clause_output, filter_model_input_emotions, filter_model_input_causes = \
        create_input_filter_model(num_docs, predicted_emotions, predicted_causes, fixed_known_emotion_cause_pair_per_doc_id, X_padded_seq)
logistic_model_per_clause_output_test, filter_model_input_emotions_test, filter_model_input_causes_test = \
        create_input_filter_model(len(X_test_padded_seq), predicted_emotions_test, predicted_causes_test, fixed_known_emotion_cause_pair_per_doc_id, X_test_padded_seq)


filter_model = logistic_filtering_model(X_padded_seq,  filter_model_input_emotions, filter_model_input_causes,
                                        logistic_model_per_clause_output, num_epochs)

fm_per_clause_predicted_output = filter_model.predict({'input_emotion': filter_model_input_emotions, 'input_cause': filter_model_input_causes})

#print(logistic_model_per_clause_output)
print(fm_per_clause_predicted_output, fm_per_clause_predicted_output.shape)

# get prediction for test.
fm_per_clause_predicted_output_predicted = filter_model.predict({'input_emotion': filter_model_input_emotions_test, 'input_cause': filter_model_input_causes_test})

print(len(fm_per_clause_predicted_output_predicted))
print(len(logistic_model_per_clause_output_test))

print(logistic_model_per_clause_output_test)
#print(fm_per_clause_predicted_output_predicted)

# smooth out the predictions
smooth_predictions_filter_model=[]
for pred in fm_per_clause_predicted_output_predicted:
  if pred>=0.5:
    smooth_predictions_filter_model.append(1)
  else:
    smooth_predictions_filter_model.append(0)
print("predictions: ")
print(smooth_predictions_filter_model)

# print accuracy, precision, recall, f1 score
print(classification_report(logistic_model_per_clause_output_test, smooth_predictions_filter_model))

"""
next TODO: 3 august 2020
1. Increase to all 800 input instead of 100 right now.
2. remove prints & push PR
3. Send PR for review to Christian
4. Find out next steps from Christian
"""
