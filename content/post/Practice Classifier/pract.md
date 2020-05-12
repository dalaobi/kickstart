---
data: 2020-05-11
title: practice classifier
---



```
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
```


```
!pip install sentencepiece
!pip install pandas
```

    Requirement already satisfied: sentencepiece in /usr/local/lib/python3.6/dist-packages (0.1.86)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (1.0.3)
    Requirement already satisfied: numpy>=1.13.3 in /usr/local/lib/python3.6/dist-packages (from pandas) (1.18.4)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas) (2018.9)
    Requirement already satisfied: python-dateutil>=2.6.1 in /usr/local/lib/python3.6/dist-packages (from pandas) (2.8.1)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.6.1->pandas) (1.12.0)



```
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

import tokenization

```


```
import pandas
```


```
def bert_encode(texts, tokenizer, max_len=256):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=256):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def bert_encode_build(texts, tokenizer, bert_layer, tag, max_len=256):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    if tag == 1:
        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    else:
        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
        input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

        _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
        clf_output = sequence_output[:, 0, :]
        out = Dense(1, activation='sigmoid')(clf_output)
        
        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
        model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
        return model

```


```
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
```


```
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
submission = pd.read_csv("./sample_submission.csv")
```


```
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
```


```
train_input = bert_encode_build(train.text.values, tokenizer, bert_layer, 1, max_len=256)
test_input = bert_encode_build(test.text.values, tokenizer, bert_layer, 1, max_len=256)
train_labels = train.target.values
```


```
model = bert_encode_build(test.text.values, tokenizer, bert_layer, 0, max_len=256)
model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_word_ids (InputLayer)     [(None, 256)]        0                                            
    __________________________________________________________________________________________________
    input_mask (InputLayer)         [(None, 256)]        0                                            
    __________________________________________________________________________________________________
    segment_ids (InputLayer)        [(None, 256)]        0                                            
    __________________________________________________________________________________________________
    keras_layer (KerasLayer)        [(None, 1024), (None 335141889   input_word_ids[0][0]             
                                                                     input_mask[0][0]                 
                                                                     segment_ids[0][0]                
    __________________________________________________________________________________________________
    tf_op_layer_strided_slice (Tens [(None, 1024)]       0           keras_layer[0][1]                
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1)            1025        tf_op_layer_strided_slice[0][0]  
    ==================================================================================================
    Total params: 335,142,914
    Trainable params: 335,142,913
    Non-trainable params: 1
    __________________________________________________________________________________________________



```
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint],
    batch_size=4,
    verbose=1,
)
```

    Epoch 1/3
    1523/1523 [==============================] - 764s 502ms/step - loss: 0.4409 - accuracy: 0.8061 - val_loss: 0.3821 - val_accuracy: 0.8319
    Epoch 2/3
    1523/1523 [==============================] - 765s 502ms/step - loss: 0.3152 - accuracy: 0.8772 - val_loss: 0.3769 - val_accuracy: 0.8464
    Epoch 3/3
    1523/1523 [==============================] - 689s 452ms/step - loss: 0.2178 - accuracy: 0.9177 - val_loss: 0.3995 - val_accuracy: 0.8273



```
model.load_weights('model.h5')
test_pred = model.predict(test_input)
```


```
submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)
```


```

```


```

```


```

```
