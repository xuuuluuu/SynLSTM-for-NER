from reader import  Reader
import numpy as np
import pickle
import torch
from transformers import *

from bert_serving.client import BertClient
bc = BertClient(port=8888)
# tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

'''
run the corresponding commands for different datasets:
# non-engligh datasets
bert-serving-start -pooling_layer -4 -model_dir multi_cased_L-12_H-768_A-12 -max_seq_len=NONE -cased_tokenization -num_worker=8 -port=8880 -pooling_strategy=NONE -cpu -show_tokens_to_client
# OntoNote English
bert-serving-start -pooling_layer -7 -model_dir cased_L-24_H-1024_A-16 -max_seq_len=NONE -cased_tokenization -num_worker=8 -port=8888 -pooling_strategy=NONE -cpu -show_tokens_to_client

'''


def read_parse_write(infile, outfile):
    reader = Reader()
    insts = reader.read_conll(infile, -1, True)
    f = open(outfile, 'wb')
    all_vecs = []
    q=0
    for inst in insts:

        tokens = []
        orig_to_tok_index = []# 0 - >0, 1-> len(all word_piece)
        for i, word in enumerate(inst.input.words):
            orig_to_tok_index.append(len(tokens))
            # word_tokens = tokenizer.tokenize(word.lower())
            word_tokens = tokenizer.tokenize(word)
            for sub_token in word_tokens:
                # orig_to_tok_index.append(i)
                tokens.append(sub_token)
        vec = bc.encode([tokens], show_tokens=True, is_tokenized=True)
        vec = vec[0][:, 1:, :][:, orig_to_tok_index, :]
        if q ==0:
            print(inst.input.words)
            print(vec[0,-1, -1])
        all_vecs.append(vec)
        q =q+1
    pickle.dump(all_vecs, f)
    f.close()

    
dep = '.sd'
dataset = 'ontonotes'
file = "data/"+dataset+"/train"+dep+".conllx"
outfile = file + ".bert." + "vec"
read_parse_write(file, outfile)
# file = "data/"+dataset+"/test"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/dev"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)


# dep = '.sd'
# dataset = 'catalan'
# file = "data/"+dataset+"/train"+dep+".conllx"
# outfile = file + ".bert." + ".vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/test"+dep+".conllx"
# outfile = file + ".bert." + ".vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/dev"+dep+".conllx"
# outfile = file + ".bert." + ".vec"
# read_parse_write(file, outfile)

# dep = '.sd'
# dataset = 'spanish'
# file = "data/"+dataset+"/train"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/test"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/dev"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)

# dep = '.sd'
# dataset = 'ontonotes_chinese'
# file = "data/"+dataset+"/train"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/test"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/dev"+dep+".conllx"
# outfile = file + ".bert." + "vec"
# read_parse_write(file, outfile)
