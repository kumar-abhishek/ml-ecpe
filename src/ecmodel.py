from tensorflow import keras

def first_layer_model(X_padded_seq, tokenizer_emotion, y_emotion_tokenized, y_cause, num_epochs):
    global model, history
    clauses_model = keras.layers.Input(shape=X_padded_seq.shape[1:], name='clause_input')
    clauses_embeddings = keras.layers.Embedding(input_length=30, input_dim=1000, output_dim=50)(clauses_model)
    output1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(clauses_embeddings)
    attention_output = keras.layers.Attention()(
        [output1, output1, output1])  # no idea why we use output3, 3 times TODO!!!
    output_emotion = keras.layers.Bidirectional(keras.layers.LSTM(64))(attention_output)
    output_emotion_dense1 = keras.layers.Dense(128, activation="relu")(output_emotion)
    output_emotion_dense2 = keras.layers.Dense(len(tokenizer_emotion.word_index) + 1, activation='softmax',
                                               name='emotion_output')(output_emotion_dense1)
    output_cause = keras.layers.Bidirectional(keras.layers.LSTM(64))(attention_output)
    output_cause_dense1 = keras.layers.Dense(128, activation="relu")(output_cause)
    output_cause_dense2 = keras.layers.Dense(1, activation='sigmoid', name='cause_output')(output_cause_dense1)
    model = keras.Model(inputs=clauses_model, outputs=[output_emotion_dense2, output_cause_dense2])
    model.compile(loss={'emotion_output': 'sparse_categorical_crossentropy', 'cause_output': 'binary_crossentropy'},
                  optimizer="adam", loss_weights=[0.5, 0.5], metrics=['accuracy'])
    print(len(X_padded_seq), len(y_emotion_tokenized), len(y_cause))
    print(y_emotion_tokenized[:5])
    print(y_cause[:5])
    print(clauses_embeddings.shape)
    # keep number of epochs >=15 so that accuracy of emotion output is decent
    history = model.fit({'clause_input': X_padded_seq}, {'emotion_output': y_emotion_tokenized, 'cause_output': y_cause},
                        epochs=num_epochs) # change epoch back to 100
    # pyplot.plot(history.history['emotion_output_accuracy'], label='emotion_output_accuracy')
    # pyplot.plot(history.history['cause_output_accuracy'], label='cause_output_accuracy')
    # pyplot.legend()
    # pyplot.show()

    # save model and architecture to single file
    # model.save('model.h5')
    # print("Saved model to disk")
    # print(X_padded_seq)
    # print(clause_global[:30])
    # cause_
    # print('causes: ', model.predict({'clause_input': X_padded_seq})[1][0:30])
    # print('-----')
    # print('emotions: ', model.predict({'clause_input': X_padded_seq})[0][1])
    return model