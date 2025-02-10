# mgt_classifier
Classifier to classify AI vs Human generated text

# Instructions to prepare data
Different llms are being used to create features using token logits and log_probs of the tokens. To calculate binocular score, perplexity, coefficeint of variation of texts, and log_probs two llms `openai-community/gpt2`, `openai-community/gpt2-medium`, and for fast_detectgpt score `EleutherAI/gpt-neo-2.7B` are used. All the inferencing was done on free google collab notebooks.
`python prepare_data.py` will create a csv file `prepared_beemo.csv` file in the `all_data` folder.
`python train.py` will train the model on `prepared_beemo.csv`
`python predict.py` will predict if a string is ai or human generated. string is passed inside the script by setting the value of `sample_string`.

To keep the scripts simple and to keep focus on the goal of creating a simple classifier, functionality of passing parameters to the scripts have not been added now.
