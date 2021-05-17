# Better Feature Integration for Named Entity Recognition

[NAACL 2021] [Better Feature Integration for Named Entity Recognition (In NAACL 2021)](https://arxiv.org/abs/2104.05316)

# Requirement
Python 3.7

Pytorch 1.4.0

Transformers 3.3.1

CUDA 10.1, 10.2

[Bert-as-service](https://github.com/hanxiao/bert-as-service)


# Running   

Firstly, download the embedding files: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/) , [cc.ca.300.vec, cc.es.300.vec, cc.zh.300.vec](https://fasttext.cc/docs/en/crawl-vectors.html), and put the files in the data folder.

By default, the model eval our saved model (without BERT) on SemEval 2010 Task 1 Spanish dataset.  

```
python main.py  
```

To train the model with other datasets:    
```
python main.py --mode=train --dataset=ontonotes --embedding_file=glove.6B.100d.txt
```

To train with BERT, first obtain the contextual embedding with the instructions in the **get_context_emb** folder (The contextual embedding files for OntoNotes Engligh can be downloaded from [***here***](https://drive.google.com/drive/folders/1Eh3RR7QDmrjUhY6MCy7QlAcXPQrRC7Fy).), and then run with the command:
```
python main.py --mode=train --dataset=ontonotes --embedding_file=glove.6B.100d.txt --context_emb=bert 
```

Note that the flag **--dep_model=dggcn** (by default) is where we call both GCN and our Syn-LSTM model. The flag **--num_lstm-layer** is designed for running some baselines, and should be set to 0 (by default) when testing our model. 

# About Dataset

Note that we use the data from 4 columns: word, dependency head index, dependency relation label, and entity label.


# Related Repo
The code are created based on [the code](https://github.com/allanj/ner_with_dependency) of the paper "Dependency-Guided LSTM-CRF Model for Named Entity Recognition", EMNLP 2019.



