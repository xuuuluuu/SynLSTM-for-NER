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
bert-serving-start -pooling_layer -1 -model_dir multi_cased_L-12_H-768_A-12 -max_seq_len=NONE -cased_tokenization -num_worker=8 -port=8880 -pooling_strategy=NONE -cpu -show_tokens_to_client
bert-serving-start -pooling_layer -1 -model_dir cased_L-24_H-1024_A-16 -max_seq_len=NONE -cased_tokenization -num_worker=8 -port=8888 -pooling_strategy=NONE -cpu -show_tokens_to_client

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
        # print(inst.input.words)
        # print(tokens)
        # print(orig_to_tok_index)
        # print(vec[0].shape)
        # print(vec[0])
        # print(vec[1])
        vec = vec[0][:, 1:, :][:, orig_to_tok_index, :]
        # print(vec)
        # input('')
        if q ==0:
            print(inst.input.words)
            print(vec[0,-1, -1])
        all_vecs.append(vec)
        q =q+1
    pickle.dump(all_vecs, f)
    f.close()

# dep = ''
# dataset = 'conll2003'
# file = "data/"+dataset+"/train"+dep+".txt"
# outfile = file + ".bert." + ".vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/test"+dep+".txt"
# outfile = file + ".bert." + ".vec"
# read_parse_write(file, outfile)
# file = "data/"+dataset+"/dev"+dep+".txt"
# outfile = file + ".bert." + ".vec"
# read_parse_write(file, outfile)


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



# from transformers import *
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bc = BertClient()
# sent = ['a', 'well-truck', 'header', 'disappoint']

# tokens = []
# orig_to_tok_index = []# 0 - >0, 1-> len(all word_piece)
# for i, word in enumerate(sent):
#     orig_to_tok_index.append(len(tokens))
#     word_tokens = tokenizer.tokenize(word)
#     for sub_token in word_tokens:
#         # tok_to_orig_index.append(i)
#         tokens.append(sub_token)

# print(tokens)
# print(orig_to_tok_index)
# res = bc.encode([tokens], show_tokens=True, is_tokenized=True)

# print(res[0].shape)
# print(res[1])
# print(res[0][:, 1:, :][:, orig_to_tok_index, :].shape)