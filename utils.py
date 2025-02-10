import numpy as np
import os
import pandas as pd
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from typing import Union

# ai vocab list
AI_VOCAB_LIST = ["provide a valuable insight", "left an indelible mark", "play a significant role in shaping", "an unwavering commitment", "open a new avenue", "a stark reminder", "play a crucial role in determining", "finding a contribution", "crucial role in understanding", "finding a shed light", "gain a comprehensive understanding", "conclusion of the study provides", "a nuanced understanding", "hold a significant", "gain significant attention", "continue to inspire", "provide a comprehensive overview", "finding the highlight the importance", "endure a legacy", "mark a significant", "gain a deeper understanding", "the multifaceted nature", "the complex interplay", "study shed light on", "need to fully understand"]

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def load_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,).eval()
    return tokenizer, model

def log_probability(encoding, logits, temperature=1):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_inputs = encoding.input_ids[..., 1:].unsqueeze(-1).contiguous()

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(shifted_logits, dim=2, index=shifted_inputs).squeeze(-1)
    return token_log_probs.to("cpu").float().numpy()

def find_phrase_occurrences(text, phrases=AI_VOCAB_LIST):
    phrase_counts = {}
    for phrase in phrases:
        # Use regex to find overlapping matches
        occurrences = len(re.findall(rf'\b{re.escape(phrase)}\b', text, re.IGNORECASE))
        phrase_counts[phrase] = occurrences
    return phrase_counts

def calculate_burstiness(text, window_size=100, overlap=50):
    # Preprocess text
    words = re.findall(r'\w+', text.lower())

    # Calculate global word frequencies
    global_freq = Counter(words)

    # Create sliding windows
    windows = []
    start = 0
    while start < len(words):
        windows.append(words[start:start + window_size])
        start += window_size - overlap

    # Calculate local frequencies for each window
    local_frequencies = []
    for window in windows:
        local_freq = Counter(window)
        local_frequencies.append(local_freq)

    # Calculate burstiness metrics
    word_burstiness = {}
    for word in global_freq:
        if global_freq[word] >= 5:  # Only consider words that appear enough times
            # Get frequency in each window
            window_freqs = [freq[word]/len(window) if len(window) > 0 else 0
                          for freq, window in zip(local_frequencies, windows)]

            # Calculate metrics
            mean_freq = np.mean(window_freqs)
            std_freq = np.std(window_freqs)

            # Burstiness score: coefficient of variation
            burstiness_score = std_freq / mean_freq if mean_freq > 0 else 0

            word_burstiness[word] = {
                'burstiness_score': burstiness_score,
                'mean_frequency': mean_freq,
                'std_frequency': std_freq
            }

    # Calculate overall text burstiness
    overall_burstiness = np.mean([stats['burstiness_score']
                                 for stats in word_burstiness.values()])

    return {
        'overall_burstiness': overall_burstiness,
        'word_burstiness': word_burstiness
    }

def get_bursty_words(burstiness_results, threshold=1.0):
    bursty_words = [
        (word, stats['burstiness_score'])
        for word, stats in burstiness_results['word_burstiness'].items()
        if stats['burstiness_score'] >= threshold
    ]
    return sorted(bursty_words, key=lambda x: x[1], reverse=True)

# fast-detect-gpt utils
def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def get_fast_detectgpt_score(list_text, scoring_tokenizer, scoring_model):
    device = scoring_model.device
    tokenized = scoring_tokenizer(list_text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(device)
    labels = tokenized.input_ids[:, 1:]
    with torch.no_grad():
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        logits_ref = logits_score
        crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        del logits_score, logits_ref
        torch.cuda.empty_cache()
    return crit

# binocular utils
def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor,
               sentences=False,
               median: bool = False,
               temperature: float = 1.0):
    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

    else:
        _, n_tokens = shifted_labels.shape
        if sentences:
            ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                   shifted_attention_mask).cumsum(dim=-1)/ torch.arange(1, n_tokens+1).to(logits.device)
            ppl = ppl.to("cpu").float().numpy()
        else:
            ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
                   shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

    return ppl

def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits / temperature, q_logits / temperature

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    if sample_p:
        p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    if median:
        ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
        agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce

def coe_of_var(model, tokenizer, paragraph, verbose=False):
    device = model.device
    # split paragraph into sentences
    sentences = re.split(r'[.!?]', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    batch_size = len(sentences)
    encodings = tokenizer(sentences,return_tensors="pt",padding="longest" if batch_size > 1 else False, truncation=True,return_token_type_ids=False).to(device)
    logits = model(**encodings.to(device)).logits
    perplexities = perplexity(encodings, logits, sentences=True)

    if verbose:
        for sent, ppl in zip(sentences, perplexities):
            print(f"Sentence: {sent}\nPerplexity: {ppl:.4f}\n")
        
    mean_ppl = np.mean(perplexities)
    std_ppl = np.std(perplexities)
    
    # Compute coefficient of variation.
    coe_of_var = std_ppl / mean_ppl if mean_ppl != 0 else 0
    return coe_of_var, mean_ppl, perplexities


class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   device_map={"": DEVICE_1},
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.bfloat16 if use_bfloat16
                                                                   else torch.float32,
                                                                   token=huggingface_config["TOKEN"]
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    device_map={"": DEVICE_2},
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16
                                                                    else torch.float32,
                                                                    token=huggingface_config["TOKEN"]
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        torch.cuda.empty_cache()
        del observer_logits, performer_logits
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred

def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


def get_probs_feature(log_prob1, log_prob2):
    assert (log_prob1.shape == log_prob2.shape) and (len(log_prob1.shape) == 2), "Log prob shapes are different"
    list_of_features = []
    add,sub,mul,div = (log_prob1 + log_prob2), (log_prob1 - log_prob2), (log_prob1 * log_prob2), (log_prob1 / log_prob2)
    list_of_features.append(np.mean(add))
    list_of_features.append(np.mean(sub))
    list_of_features.append(np.mean(mul))
    list_of_features.append(np.mean(div))
    list_of_features.append(np.std(add))
    list_of_features.append(np.std(sub))
    list_of_features.append(np.std(mul))
    list_of_features.append(np.std(div))
    list_of_features.append(log_prob1.shape[1])
    list_of_features.append(LA.norm(add))
    return list_of_features