Install the bert-as-service

Edit the dataset path in the bas.py

Running the commands to start bert-as-service
```
bert-serving-start -pooling_layer -7 -model_dir cased_L-24_H-1024_A-16 -max_seq_len=NONE -cased_tokenization -num_worker=8 -port=8888 -pooling_strategy=NONE -cpu -show_tokens_to_client
```

Then run the command to geneter the context embedding:
```
python bas.py
```

Double-check the name of the generated files, and move them to the correpsoinding sub-folders under ***data*** folder.

