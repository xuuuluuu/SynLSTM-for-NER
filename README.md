# SynLSTM-for-NER

[NAACL 2021] [SynLSTM-for-NER (In NAACL 2021)]()

# Requirement
Python 3.7

Pytorch 1.4.0 or above

Transformers 3.3.1

CUDA 10.1, 10.2

Bert-as-service


# Running   
```
python main.py  
```
By default, the model eval our saved model (without BERT) on SemEval 2010 Task 1 Catalan dataset.  

To train the model with other datasets:    
```
python main.py --mode=train --dataset=ontonotes
```

To run with BERT, first download the saved BERT features in the link, and then run with the command:
```
python main.py --mode=train --context_emb=bert
```


# Related Repo
The code are created based on [the code](https://github.com/allanj/ner_with_dependency) of the paper "Dependency-Guided LSTM-CRF Model for Named Entity Recognition", EMNLP 2019.

