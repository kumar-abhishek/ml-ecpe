import numpy as np
from tensorflow import keras
from collections import defaultdict
from create_dataset import *


def create_input_filter_model(num_docs, predicted_emotions, predicted_causes, fixed_known_emotion_cause_pair_per_doc_id, X_padded_seq):
    #global i, ec_pair_per_doc_id, cause, emotion, item
    cause_clauses_per_doc_id = defaultdict(list)
    emotion_clauses_per_doc_id = defaultdict(list)
    doc_id_to_clause_id_emotions_map = defaultdict(list)
    for clause_id, cause_prob in enumerate(predicted_causes):
        if cause_prob > 0.5:
            cause_clauses_per_doc_id[clauseid_to_docid[clause_id]].append(clause_id)
    # print('cause_clauses_per_doc_id: ', cause_clauses_per_doc_id)
    for clause_id, emotion_prob in enumerate(predicted_emotions):
        max_emotion_prob = 0.0
        max_idx = 0
        for i, prob in enumerate(emotion_prob):
            if prob > max_emotion_prob:
                max_emotion_prob = prob
                max_idx = i

        #max_emotion = tokenized_id_to_emotion_map[max_idx]
        if max_idx != 1:  # skip 'xxxxxx' as emotion(unknown)
            emotion_clauses_per_doc_id[clauseid_to_docid[clause_id]].append(max_idx)
            doc_id_to_clause_id_emotions_map[clauseid_to_docid[clause_id]].append(clause_id)
    print('emotion_clauses_per_doc_id:', emotion_clauses_per_doc_id)
    print('cause_clauses_per_doc_id: ', cause_clauses_per_doc_id)
    print('doc_id_to_clause_id_emotions_map:', doc_id_to_clause_id_emotions_map)

    # input : emotion * cause
    emotion_cause_pair_per_doc_id = []
    ec_pair_per_doc_id = []
    for doc_id in range(num_docs):
        causes = cause_clauses_per_doc_id[doc_id]  # predicted causes
        emotions = emotion_clauses_per_doc_id[doc_id]  # predicated emotions
        emotion_clause_ids = doc_id_to_clause_id_emotions_map[doc_id]
        emotion_cause_pair = []  # why is this needed?
        ec_pair = []  # ec_pair is (emotion clause id, cause clause id) pairs

        for cause in causes:
            for emotion in emotions:
                if emotion != 1:
                    emotion_cause_pair.append((emotion, cause))

        for cause in causes:
            for e in emotion_clause_ids:
                if cause == e:  # cause_id == emotion_id
                    print('cause==e', cause)
                    continue
                ec_pair.append((e, cause))

        emotion_cause_pair_per_doc_id.append(emotion_cause_pair)
        ec_pair_per_doc_id.append(ec_pair)
    ppp = []  # created only for nicer output
    for item in emotion_cause_pair_per_doc_id:
        for inner_item in item:
            ppp.append(list(inner_item))
    print('emotion_cause_pair_per_doc_id: (tokenized_emotion, cause_clause_id)  :', ppp)
    print('fixed_known_emotion_cause_pair_per_doc_id:                           :', fixed_known_emotion_cause_pair_per_doc_id)
    print('ec_pair_per_doc_id: (emotion_clause_id, cause_clause_id)             :', ec_pair_per_doc_id)
    print(len(emotion_cause_pair_per_doc_id), len(fixed_known_emotion_cause_pair_per_doc_id), len(ec_pair_per_doc_id))
    for i in range(num_docs):
        print("emotion_cause_pair_per_doc_id:             ", emotion_cause_pair_per_doc_id[i])
        print('fixed_known_emotion_cause_pair_per_doc_id:  ', fixed_known_emotion_cause_pair_per_doc_id[i])
        print('------------------------------------------')
    logic_model_output = []
    for doc_id, emotion_cause_pairs in enumerate(emotion_cause_pair_per_doc_id):
        output_per_doc_id = []  # output_per_doc_id is a list of whether expected emotion_cause_pair matches with predicted pair.
        # print(doc_id, emotion_cause_pairs)
        for pair in emotion_cause_pairs:
            if pair[0] == fixed_known_emotion_cause_pair_per_doc_id[doc_id][0] and pair[1] == \
                    fixed_known_emotion_cause_pair_per_doc_id[doc_id][1]:
                output_per_doc_id.append(1)
            else:
                output_per_doc_id.append(0)
        logic_model_output.append(output_per_doc_id)
    logic_model_output = np.asarray(logic_model_output)
    print('logic_model_output: ', logic_model_output[:100], logic_model_output.shape)
    #return logic_model_output, ec_pair_per_doc_id
    #global filter_model_input_emotions, filter_model_input_causes, logistic_model_per_clause_output, doc_id, idx, pair, e, filter_model
    # creating input vector
    filter_model_input_emotions = []
    filter_model_input_causes = []
    logistic_model_per_clause_output = []
    empty_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # len 30
    for doc_id, pairs in enumerate(ec_pair_per_doc_id):  # ec_pair is (emotion clause id, cause clause id) pairs
        if not pairs:  # pairs is ec_pair
            filter_model_input_emotions.append(empty_list)
            filter_model_input_causes.append(empty_list)
            logistic_model_per_clause_output.append(1)
        else:
            output_list = logic_model_output[doc_id]
            print(output_list, pairs, doc_id)
            for idx, pair in enumerate(pairs):
                (e, c) = pair  # e is emotion_clause_id, c is cause_clause_id
                # print(idx, pair, e, c)
                filter_model_input_emotions.append(X_padded_seq[e].tolist())
                filter_model_input_causes.append(X_padded_seq[c].tolist())
                if idx < len(output_list):  # Why?
                    logistic_model_per_clause_output.append(output_list[idx])
    filter_model_input_emotions = np.asarray(filter_model_input_emotions)
    filter_model_input_causes = np.asarray(filter_model_input_causes)
    filter_model_input_emotions = np.array(list(x for x in filter_model_input_emotions))
    filter_model_input_causes = np.array(list(x for x in filter_model_input_causes))
    logistic_model_per_clause_output = np.array(list(x for x in logistic_model_per_clause_output))
    print(filter_model_input_emotions[:5], filter_model_input_emotions.shape)
    print('-------\n')
    print(filter_model_input_causes[0:5], filter_model_input_causes.shape)
    print('-------\n')
    print(logistic_model_per_clause_output[0:180], type(logistic_model_per_clause_output))
    print(X_padded_seq.shape)
    #print(y_cause, type(y_cause), y_cause.shape)
    print('filter_model_input_emotions.shape: ', filter_model_input_emotions.shape)
    print('filter_model_input_causes.shape:', filter_model_input_causes.shape)
    print('logistic_model_per_clause_output.shape', logistic_model_per_clause_output.shape)
    return logistic_model_per_clause_output, filter_model_input_emotions, filter_model_input_causes


def logistic_filtering_model(X_padded_seq, filter_model_input_emotions, filter_model_input_causes, logistic_model_per_clause_output, num_epochs):

    # logistic regresson
    input_emotion = keras.layers.Input(shape=X_padded_seq.shape[1:], name='input_emotion')
    input_cause = keras.layers.Input(shape=X_padded_seq.shape[1:], name='input_cause')
    combined = keras.layers.Concatenate(axis=1)([input_emotion, input_cause])
    print('combined.shape: ', combined.shape)
    output_layer = keras.layers.Dense(1, activation='sigmoid', name='filter_model_output')(combined)
    filter_model = keras.Model(inputs=[input_emotion, input_cause], outputs=output_layer)
    filter_model.compile(loss="binary_crossentropy",
                         optimizer="sgd", metrics=['accuracy'])
    # TODO: accuracy is not increasing beyond a certain value, why?
    history = filter_model.fit({'input_emotion': filter_model_input_emotions, 'input_cause': filter_model_input_causes},
                               {'filter_model_output': logistic_model_per_clause_output},
                               # {'filter_model_output': filter_model_input_causes},
                               epochs=num_epochs, verbose=1)
    return filter_model