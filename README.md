# SynLSTM-for-NER

[NAACL 2021] [SynLSTM-for-NER (In NAACL 2021)]()

# Requirement
Python 3.7

Pytorch 1.4.0 or above

Transformers 3.3.1

CUDA 10.1, 10.2

[Bert-as-service](https://github.com/hanxiao/bert-as-service)


# Running   
```
python main.py  
```
By default, the model eval our saved model (without BERT) on SemEval 2010 Task 1 Spanish dataset.  

To train the model with other datasets (we use fasttext 300d for the Chinese, Spanish, and Catalan datasets.):    
```
python main.py --mode=train --dataset=ontonotes --embedding_file=glove.6B.100d.txt
```

To train with BERT, first obtain the contextual embedding with the instructions in the **get_context_emb** folder, and then run with the command:
```
python main.py --mode=train --context_emb=bert
```

# About Dataset

If you have the LDC license for OntoNotes, please send me the proof (e.g., screenshot) and I can share the preprocessed OntoNotes datasets.
Note that we use the data from 4 columns: word, dependency head index, dependency relation label, and entity label.


# Related Repo
The code are created based on [the code](https://github.com/allanj/ner_with_dependency) of the paper "Dependency-Guided LSTM-CRF Model for Named Entity Recognition", EMNLP 2019.



