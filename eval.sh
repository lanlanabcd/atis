#!/usr/bin/env bash
python3 run.py --raw_train_filename="../atis_data/data/resplit/processed/train_with_tables.pkl" \
               --raw_dev_filename="../atis_data/data/resplit/processed/dev_with_tables.pkl" \
               --raw_validation_filename="../atis_data/data/resplit/processed/valid_with_tables.pkl" \
               --raw_test_filename="../atis_data/data/resplit/processed/test_with_tables.pkl" \
               --input_key="utterance" \
               --anonymize=1 \
               --anonymization_scoring=1 \
               --use_snippets=0 \
               --state_positional_embeddings=1 \
               --snippet_age_embedding=1 \
               --discourse_level_lstm=1 \
               --interaction_level=1 \
               --reweight_batch=1 \
               --train=1\
               --evaluate=1 \
               --evaluate_split="valid" \
               --use_predicted_queries=1 \
<<<<<<< HEAD
               --save_file="logs/save_30" # You need to edit this.
=======
               --save_file="logs/save_20" # You need to edit this.
>>>>>>> 73ce4e9374009a50da963d0a51f14174e5bab5a8

python3 metric_averages.py "logs/devgold_predictions.json" # You need to edit this.
