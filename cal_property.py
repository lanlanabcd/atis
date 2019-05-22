import numpy as np
import pickle

data = pickle.load(open("interactions_new_train", "rb"))
cnt = 0
copy_len = 0
all_len = 0
max_len = 0
for interaction in data:
    for utterance in interaction.utterances:
        all_len += len(utterance.gold_query_to_use)
        if len(utterance.gold_query_to_use) > max_len:
            max_len = len(utterance.gold_query_to_use)
        if utterance.copy_gold_query:
            copy_len += len(utterance.copy_gold_query)
        else:
            copy_len += len(utterance.gold_query_to_use)
        cnt += 1
print("copy average length: ", float(copy_len) / cnt)
print("total average length: ", float(all_len) / cnt)
print("max langth: ", max_len)
