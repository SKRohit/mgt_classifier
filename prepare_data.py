# all imports

import os
import numpy as np
import pandas as pd
import re
import torch
import transformers

from numpy import linalg as LA
import matplotlib.pyplot as plt
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from utils import *

torch.set_grad_enabled(False)

# downlaod data
beemo_df = pd.read_parquet("hf://datasets/toloka/beemo/data/train-00000-of-00001.parquet")
print("############# SAMPLE DATA (#############")
print(beemo_df.head(3))

# prepare binocular feature
binocular_observer, binocular_performer = "openai-community/gpt2", "openai-community/gpt2-medium"
assert_tokenizer_consistency(binocular_observer, binocular_performer)
bino = Binoculars(observer_name_or_path=binocular_observer, performer_name_or_path=binocular_performer)

batch_size = 32

# create empty lists to store the scores
human_scores, model_scores = [], []

with torch.no_grad():
    # iterate through the DataFrame in batches
    for i in tqdm(range(0, len(beemo_df), batch_size)):
        torch.cuda.empty_cache()
        batch = beemo_df.iloc[i:i + batch_size]
        model_outputs = batch['model_output'].tolist()
        human_outputs = batch['human_output'].tolist()
    
        # calculate score in batches
        batch_model_scores = bino.compute_score(model_outputs)
        batch_human_scores = bino.compute_score(human_outputs)

        model_scores.extend(batch_model_scores)
        human_scores.extend(batch_human_scores)

beemo_df["binocular_human"] = human_scores
beemo_df["binocular_model"] = model_scores

# prepare cov and perplexity
m_covs_ppls, h_covs_ppls = [], []
with torch.no_grad():
    for i in tqdm(range(0, len(beemo_df))):
        torch.cuda.empty_cache()
        batch = beemo_df.iloc[i]
        model_outputs = batch['model_output']
        human_outputs = batch['human_output']
        m_cov, m_ppl, _ = coe_of_var(bino.observer_model, bino.tokenizer, model_outputs, verbose=False)
        h_cov, h_ppl, _ = coe_of_var(bino.observer_model, bino.tokenizer, human_outputs, verbose=False)
        m_covs_ppls.append([m_cov, m_ppl])
        h_covs_ppls.append([h_cov, h_ppl])

m_covs_ppls, h_covs_ppls = np.array(m_covs_ppls), np.array(h_covs_ppls)
beemo_df['cov_model'], beemo_df['ppl_model'] = m_covs_ppls[:,0], m_covs_ppls[:,1]
beemo_df['cov_human'], beemo_df['ppl_human'] = h_covs_ppls[:,0], h_covs_ppls[:,1]

# prepare log_probs feature
device1, device2 = bino.observer_model.device, bino.performer_model.device
# running with batch size 1 is faster becuase of 
batch_size = 1

combined_log_prob = []
with torch.no_grad():
    for i in tqdm(range(0, len(beemo_df), batch_size)):
        torch.cuda.empty_cache()
        batch = beemo_df.iloc[i:i + batch_size]
        model_outputs = batch['model_output'].tolist()
        human_outputs = batch['human_output'].tolist()

        # ai text log prob
        encoding1 = bino.tokenizer(model_outputs, return_tensors="pt",padding="longest" if batch_size > 1 else False, truncation=True,return_token_type_ids=False).to(device1)
        encoding2 = bino.tokenizer(model_outputs, return_tensors="pt",padding="longest" if batch_size > 1 else False, truncation=True,return_token_type_ids=False).to(device2)
        log_prob_1 = log_probability(encoding1,bino.observer_model(**encoding1).logits)
        log_prob_2 = log_probability(encoding2,bino.performer_model(**encoding2).logits)
        m_log_probs = get_probs_feature(log_prob_1, log_prob_2)
        torch.cuda.empty_cache()

        # human text log prob
        encoding1 = bino.tokenizer(human_outputs, return_tensors="pt",padding="longest" if batch_size > 1 else False, truncation=True,return_token_type_ids=False).to(device1)
        encoding2 = bino.tokenizer(human_outputs, return_tensors="pt",padding="longest" if batch_size > 1 else False, truncation=True,return_token_type_ids=False).to(device2)
        log_prob_1 = log_probability(encoding1,bino.observer_model(**encoding1).logits)
        log_prob_2 = log_probability(encoding2,bino.performer_model(**encoding2).logits)
        h_log_probs = get_probs_feature(log_prob_1, log_prob_2)
        torch.cuda.empty_cache()

        combined_log_prob.append(m_log_probs + h_log_probs)

del bino, encoding1, encoding2
torch.cuda.empty_cache()

# prepare fast_gpt_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scoring_model_name = "EleutherAI/gpt-neo-2.7B"
fast_gpt_tokenizer, fast_gpt_model = load_tokenizer_model(scoring_model_name)
fast_gpt_model.to(device)

batch_size = 1

m_fastgpt_scores, h_fastgpt_scores = [], []
with torch.no_grad():
    for i in tqdm(range(0, len(beemo_df), batch_size)):
        torch.cuda.empty_cache()
        batch = beemo_df.iloc[i:i + batch_size]
        model_outputs = batch['model_output'].tolist()
        human_outputs = batch['human_output'].tolist()
        m_fastgpt_scores.extend(get_fast_detectgpt_score(model_outputs, fast_gpt_tokenizer, fast_gpt_model))
        h_fastgpt_scores.extend(get_fast_detectgpt_score(human_outputs, fast_gpt_tokenizer, fast_gpt_model))

beemo_df["fast_detect_human"] = h_fastgpt_scores
beemo_df["fast_detect_model"] = m_fastgpt_scores

del fast_gpt_model
torch.cuda.empty_cache()

# combine all feature in dataframe
log_prob_feature_columns = ['model_add_mean', 'model_sub_mean',
       'model_mul_mean', 'model_div_mean', 'model_add_std', 'model_sub_std',
       'model_mul_std', 'model_div_std', 'model_word_length', 'model_l2norm', 'human_add_mean', 'human_sub_mean', 'human_mul_mean', 'human_div_mean',
       'human_add_std', 'human_sub_std', 'human_mul_std', 'human_div_std',
       'human_word_length', 'human_l2norm']
combined_log_prob = np.array(combined_log_prob)
for i,col in enumerate(log_prob_feature_columns):
    beemo_df[col] = combined_log_prob[:,i]

# convert beemo_df to train and test data
human_df = beemo_df[['human_output','binocular_human','fast_detect_human','cov_human','ppl_human','human_add_mean', 'human_sub_mean', 'human_mul_mean', 'human_div_mean',
       'human_add_std', 'human_sub_std', 'human_mul_std', 'human_div_std',
       'human_word_length', 'human_l2norm']]

model_df = beemo_df[['model_output','binocular_model','fast_detect_model','cov_model','ppl_model','model_add_mean', 'model_sub_mean',
       'model_mul_mean', 'model_div_mean', 'model_add_std', 'model_sub_std',
       'model_mul_std', 'model_div_std', 'model_word_length', 'model_l2norm',]]

human_df['label'] = [0]*len(human_df)
model_df['label'] = [1]*len(model_df)

columns = ['text','binocular','fast_detect','cov','ppl','add_mean','sub_mean','mul_mean','div_mean','add_std','sub_std','mul_std','div_std','word_length','l2norm','label']

human_df['label'] = [0]*len(human_df)
model_df['label'] = [1]*len(model_df)
final_df = pd.concat([human_df, model_df], ignore_index=True, sort=False).sample(frac=1).fillna(0)

print("############# PREPARED DATA (#############")
print(final_df.head(3))

final_df.to_csv("all_data/prepared_beemo.csv", index=False)