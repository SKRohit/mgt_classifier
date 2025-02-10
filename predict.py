# all imports

import numpy as np
import torch
from pickle import load

from utils import *

torch.set_grad_enabled(False)

# text to perform classification
sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

# prepare binocular feature
with torch.no_grad():
    binocular_observer, binocular_performer = "openai-community/gpt2", "openai-community/gpt2-medium"
    
    assert_tokenizer_consistency(binocular_observer, binocular_performer)
    
    bino = Binoculars(observer_name_or_path=binocular_observer, performer_name_or_path=binocular_performer)
    bino_score = bino.compute_score(sample_string)
    torch.cuda.empty_cache()
    
    # prepare coefficient of variation and perplexity
    cov, ppl, _ = coe_of_var(bino.observer_model, bino.tokenizer, sample_string, verbose=False)
    torch.cuda.empty_cache()
    
    # log_probs feature
    device1, device2 = bino.observer_model.device, bino.performer_model.device
    encoding1 = bino.tokenizer([sample_string], return_tensors="pt",padding=False, truncation=True,return_token_type_ids=False).to(device1)
    encoding2 = bino.tokenizer([sample_string], return_tensors="pt",padding=False, truncation=True,return_token_type_ids=False).to(device2)
    log_probs1 = log_probability(encoding1,bino.observer_model(**encoding1).logits)
    log_probs2 = log_probability(encoding2,bino.performer_model(**encoding2).logits)
    log_prob_features = get_probs_feature(log_probs1, log_probs2)
    
    del bino
    torch.cuda.empty_cache()
    
    # prepare fast_gpt score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_model_name = "EleutherAI/gpt-neo-2.7B"
    fast_gpt_tokenizer, fast_gpt_model = load_tokenizer_model(scoring_model_name)
    fast_gpt_model.to(device)
    fastgpt_score = get_fast_detectgpt_score([sample_string], fast_gpt_tokenizer, fast_gpt_model)
    del fast_gpt_model
    torch.cuda.empty_cache()

# final feature vector
test_feature = np.nan_to_num(np.array([bino_score, fastgpt_score, cov, ppl] + log_prob_features))
test_feature = test_feature.reshape(1,len(test_feature))
print("############# Final Feature Vector #############")
print(test_feature)

# load trained model, features_list and predict
with open("all_data/mgt_classifier.pkl", "rb") as f:
    clf2 = load(f)

with open('all_data/features_list.pkl', 'rb') as f:
    features_list = load(f)

prediction = clf2.predict(test_feature[:,features_list[0]])
ai_or_human = "AI" if prediction==1 else "Human"
print(f"\n\nText is {ai_or_human} generated.")
