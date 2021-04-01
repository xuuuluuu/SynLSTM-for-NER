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
By default, the model runs on SemEval 2010 Task 1 Catalan dataset with provided hyper-parameters without BERT.  

To run with other datasets:    
```
python main.py  --dataset=ontonotes
```

To run with BERT, first download the saved BERT features in the link, and then run with the command:
```
python main.py  --context_emb=bert
```

To run with the saved best model:  
```
python main.py  --mode=test
```
The code will automatically load the corresponding saved model of datasets.


# Related Repo
The code are created based on [this repo](https://github.com/allanj/pytorch_lstmcrf).

