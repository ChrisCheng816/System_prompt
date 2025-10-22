import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import pickle
import numpy as np
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
import nltk
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, losses, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sacrebleu
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def indv_meteor_score(dataframe):
    msdict = []
    for ref, pred in zip(dataframe['originalComment'], dataframe['codeComment']):
        ref = ref.strip()
        pred = pred.strip()
        if pred == '':
            msdict.append(0.0)
            continue
        # 分词
        ref_tokens = word_tokenize(ref)
        pred_tokens = word_tokenize(pred)
        ms = single_meteor_score(ref_tokens, pred_tokens)
        msdict.append(ms)

    dataframe['meteor'] = msdict
    return dataframe

def official_bert_score(dataframe):
	from bert_score import score

	p_bert = []
	r_bert = []
	f1_bert = []

	p, r, f1 = score(list(dataframe['codeComment']), list(dataframe['originalComment']), lang='en', rescale_with_baseline=True)
	for pscore, rscore, f1score in zip(p.numpy(), r.numpy(), f1.numpy()):
		p_bert.append(pscore)
		r_bert.append(rscore)
		f1_bert.append(f1score)


	dataframe['bert-score-precision'] = p_bert
	dataframe['bert-score-recall'] = r_bert
	dataframe['bert-score-f1'] = f1_bert
	return dataframe


def codet5_plus_encoding(dataframe):
	
	# Mean Pooling - Take attention mask into account for correct averaging
	def mean_pooling(model_output, attention_mask):
	    # First element of model_output contains all token embeddings
	    token_embeddings = model_output[0]
	    input_mask_expanded = attention_mask.unsqueeze(
	        -1).expand(token_embeddings.size()).float()
	    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
	
	checkpoint = "Salesforce/codet5p-220m"
	device = "cuda"

	tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

	similarities = []
	for idx,item in tqdm(dataframe.iterrows()):
		sentences = []
		sentences.append(item['codeFunctions'])
		sentences.append(item['codeComment'])

		encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)

		model_output = model.encoder(
			input_ids=encoded_input["input_ids"], 
			attention_mask=encoded_input["attention_mask"], 
			return_dict=True
		)
	  
		# Perform pooling
		sentence_embeddings = mean_pooling(
			model_output, encoded_input['attention_mask'])

		# Normalize embeddings
		sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

		sim = util.pytorch_cos_sim(
			sentence_embeddings[0], sentence_embeddings[1]).item()

		similarities.append(sim)
	
	dataframe['CodeT5-plus-cosine-similarity'] = similarities
	return dataframe

def smart_join(arr):
    result = ''
    for i, word in enumerate(arr):
        if word in {'.', ',', '?', '!', ';', ':'}:
            result = result.rstrip() + word  # 去掉末尾空格再加标点
        elif word == '(':
            # 左括号前要保留空格，但后面不加空格
            result += word
        elif word == ')':
            # 右括号前去掉空格，不加空格
            result = result.rstrip() + word + ' '
        else:
            result += word + ' '
    return result.strip()

def main():
    java_dataset = load_dataset("google/code_x_glue_ct_code_to_text", "java")["test"]
    python_dataset = load_dataset("google/code_x_glue_ct_code_to_text", "python")["test"]
    root_dir = "../summarization_results"
    codes_java = [smart_join(tokens) for tokens in java_dataset["code_tokens"]]
    codes_python = [smart_join(tokens) for tokens in python_dataset["code_tokens"]]
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "openai" not in dirpath :
            continue
        codes = codes_java if "java" in dirpath else codes_python
        preds = []
        refs = []
        if "predictions_cleaned.jsonl" in filenames:
            prefile = "predictions_cleaned.jsonl"
        elif "predictions.jsonl" in filenames:
            prefile = "predictions.jsonl"
        else:
            continue
        if prefile in filenames and "references.txt" in filenames:
            pre_path = os.path.join(dirpath, prefile)
            ref_path = os.path.join(dirpath, "references.txt")
            with open(pre_path, "r", encoding="utf-8") as f:
                for line in f:
                    preds.append(json.loads(line)["prediction"])
            with open(ref_path, "r", encoding="utf-8") as f:
                for line in f:
                    # 按第一个 tab 或空格切分
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        idx, text = parts
                        refs.append(text.strip())
                    else:
                        raise ValueError(f"参考文件格式错误: {line.strip()}")
            df = pd.DataFrame({
                "originalComment": refs,
                "codeComment": preds,
                "codeFunctions": codes
            })
            print(f"References length is {len(refs)}")
            print(f"Predictions length is {len(preds)}")
            df = codet5_plus_encoding(df)
            df = official_bert_score(df)
            df = indv_meteor_score(df)
            print(f"Task stored in {dirpath} has been finished")
            metric_path = os.path.join(dirpath, "metrics_results.jsonl")
            df.to_json(metric_path, orient="records", lines=True, force_ascii=False)

            output_path = os.path.join(dirpath, "output.json")

            # 如果文件存在，先读取原数据
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    output_data = json.load(f)
                summary_stats = {
                    "num_samples": len(df),
                    "CodeT5_plus_cosine_mean": df["CodeT5-plus-cosine-similarity"].mean().item(),
                    "CodeT5_plus_cosine_median": df["CodeT5-plus-cosine-similarity"].median().item(),
                    "BERTScore_F1_mean": df["bert-score-f1"].mean().item(),
                    "BERTScore_F1_median": df["bert-score-f1"].median().item(),
                    "METEOR_mean": df["meteor"].mean().item(),
                    "METEOR_median": df["meteor"].median().item()
                }
                output_data["summary"] = summary_stats

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
	main()