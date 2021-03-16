#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path.append("../")

import urban_dictionary_scraper
import torch
import re

import pickle
import wiki_article
import dictionary_definition
import glob
import modeling
import itertools
import random
import pandas as pd
import numpy as np
import datasets
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from io import StringIO
from ipywidgets import interact, interactive, fixed, interact_manual
from transformers import AutoModelWithLMHead, AutoTokenizer
from scipy import stats
import hashlib
from collections import OrderedDict
from types import SimpleNamespace

def get_checkpoints(base_dir):
    checkpoint_dirs = glob.glob(f"{base_dir}/checkpoint*")
    checkpoint_dirs.sort(key=lambda x: int(x[(x.index("checkpoint-") + len("checkpoint-")):]))
    return checkpoint_dirs
modeling_gpt
def evaluate_lm_checkpoints(base_dir, validation_path):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    for d in get_checkpoints(base_dir):
        model = AutoModelWithLMHead.from_pretrained(d).to('cuda')
        refined_model_eval = wiki_article.lm_eval(model, tokenizer, validation_path)
        print(f"{d}: {refined_model_eval}")
tokenizer
def evaluate_title_checkpoints(base_dir, validation_path):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")print(parsed_urban_dictionary_scraperpage.body.prettify())
    for d in get_checkpoints(base_dir):
        model = AutoModelWithLMHead.from_pretrained(d).to('cuda')
        refined_model_eval = wiki_article.run_title_evaluation(model, tokenizer, validation_path)
        print(f"{d}: m={refined_model_eval.mean}, v={refined_model_eval.variance}")

# evaluate_lm_checkAutoModelWithLMHead, AutoTokenizer, points("models/wikitext_103_stride_512_v0/", "data/wikitext-103-title-train/wiki_title.valid.raw")
#print(glob.glob("models/wikitext_103_stride_512_v0/*"))

with open(f"data/en_dictionary_parsed_randomized.pickle", "rb") as f:
    parsed_dictionary = pickle.load(f)

potential_blacklist = set()
for word in parsed_dictionary:
    potential_blacklist.add(word.word)
    potential_blacklist.update(word.derivatives)
print(len(parsed_dictionary))
print(len(potential_blacklist))

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
args = SimpleNamespace()
args.block_size = 768
dataset = datasets.ParsedDictionaryDefinitionDataset(tokenizer, args, None, None, None)

flattened_set = list(itertools.chain.from_iterable(dataset._make_examples(tokenizer, e) for e in parsed_dictionary))

# print(f"{len(flattened_set)} from {len(parsed_dictionary)} entries")
word = tokenizer.encode("vitellogenin")
print(tokenizer.decode(dataset.bos_token_ids  + [1] + dataset.eos_token_ids))
print(tokenizer.decode(tokenizer.encode("<|bod|>\"<|eod|>")))

print(f"\"{tokenizer.decode(dataset.pos_sep_ids)}\"")
tokenizer.decode(dataset._make_examples(tokenizer, parsed_dictionary[0])[0])
# for example in random.choices(flattened_set, k=20):
#     print(tokenizer.decode(example))

for example in dataset._make_examples(tokenizer, parsed_dictionary[10430]):
    print(tokenizer.decode(example))

with open("data/all_words.pickle", "rb") as f:
    #words = pickle.load(f)
    #items = list(words.items())
    random.shuffle(items)
    items = OrderedDict(items)

with open("data/all_words_randomized.pickle", "wb") as f:
    pickle.dump(items, f, pickle.HIGHEST_PROTOCOL)

urban_dictionary_scraper.UrbanDictionaryDataset._make_examples(tokenizer, words[2])

model = AutoModelWithLMHead.from_pretrained("gpt2").to('cuda')

unrefined_model_eval = wiki_article.run_title_evaluation(urban_dictionary_scrapermodel, tokenizer, "wikitext-103-raw/wiki.valid.raw")
unrefined_model_eval

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("output_103/").to('cuda')

refined_model_eval = wiki_article.run_title_evaluation(model, tokenizer, "wikitext-103-raw/wiki.valid.raw")
refined_model_eval

sequence = f"\"TITLE\" is a song collaboration by Chinese artist Pamela Chen and Canadian singer Thomas Dimson, first released independently in March 2020. After gaining popularity amongst the cat community, the single was re-released by major label Columbia Records in May 2020. Pamela describes the song as being originally inspired by her two kittens, Apollo and Bean who once said meow.<bot>"

model =  modeling.GPT2LMHeadWithWeightedLossModel.from_pretrained("models/wikitext-103-raw-title-scale-20-lr5e-5").to("cuda")
input = tokenizer.encode(sequence, return_tensors="pt").to('cuda')
generated = model.generate(input, max_length=100, num_return_sequences=100, temperature=1)

print(f"Prompt text: {sequence}")
for i in range(generated.size()[0]):
    sentence_tokens = generated[i, :].tolist()
    decoded = tokenizer.decode(sentence_tokens)
    m = re.search(r"<bot>(.*?)<eot>", decoded)
    if m:urban_dictionary_scraper
        print(f"{i}) {m.groups(1)}")
    else:
        print(f"{i}) Didn't work")

resulting_string = tokenizer.decode(generated.tolist()[0])
# print(resulting_string)

for entry in entries:
    m = re.match(r"\s*" + re.escape(entry.title) + r"\d*\s*(\|[^|]*\|)?\s*", entry.entry_str)
    if m:
        trainable_entry = entry.entry_str[m.span()[1]:].strip()
        if not trainable_entry:
            raise RuntimeError(f"Bad entry for {entry.title}: '{entry.entry_str}'")
    else:
        raise RuntimeError(f"Couldn't match {entry.title} on '{entry.entry_str}'")

dictionary_path = "data/com_apple_MobileAsset_DictionaryServices_dictionaryOSX/69b7ab1cf0f75ad16bf6662b0a77fbfd36b7941f.asset/AssetData/New Oxford American Dictionary.dictionary/Contents/Resources/Body.data"
with open(dictionary_path, "rb") as f:
    valid_words = {e.title.upper() for e in dictionary_definition.DictionaryDefinition.gen_from_apple_dictionary(f)}full_dataset = [

]

model =  modeling.GPT2LMHeadWithWeightedLossModel.from_pretrained("models/dictionary-scale-10-lr5e-5").to("cuda")

words = dictionary_definition.generate_words(
    tokenizer, model, allow_proper_nouns=False, blacklist=valid_words, num=1000, max_iterations=40
)
words.sort(key=lambda x: x.title)
for w in words:
    print(f"{w} {w.entry_str}")

with open("words.tsv", "w") as f:
    for word in words:
        f.write(f"{word.title}\t{word.entry_str}\n")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
blacklist = set((x.lower() for x in itertools.chain.from_iterable(
    [e.word] + e.derivatives
    for e in pickle.load(open(f"data/en_dictionary_parsed_randomized.pickle", "rb")))
))
model = AutoModelWithLMHead.from_pretrained("models/en_dictionary_parsed_lr_00001/checkpoint-120000").to("cuda:0")

def print_words(words, f):
    for word in words:
        word_str = [word.word]
        if word.pos:
            word_str.append(f"/{word.pos}/")
        if word.topic:
            word_str.append(f"[{word.topic}]")
        print(" ".join(word_str), file=f)
        print(f"\t{word.definition}", file=f)
        print(f"\t\"{word.example}\"{' |e|' if word.from_example_expansion else ''}", file=f))

        print("", file=f)

words.sort(key=lambda x: x.word)
with open("words_with_examples.txt", "w") as f:
    print_words(words, f)

words, stats = datasets.ParsedDictionaryDefinitionDataset.generate_words(
    tokenizer, model,
    num=500,
    max_iterations=40,
    blacklist=blacklist,
    do_example_expansion=True,
    generation_args=dict(
        top_k=300,
        num_return_sequences=100,
        max_length=512,
        do_sample=True,
    ),
    expansion_generation_overrides=dict(
        top_k=50,
        num_return_sequences=10,
        do_sample=True,
    ),
    num_expansion_candidates=10,
    filter_proper_nouns=True,
)

print(stats)
print()
print_words(words, sys.stdout)

# from datasets import SpecialTokens
# """
# input_str = f"{tokenizer.bos_token}"
# input_str = "<|bod|>corner<|pos|>noun<|bd|>a point or space in a hierarchy that is within the order to which it moves along the axis.<|eod|>"
# input = tokenizer.encode(input_str, return_tensors="pt").to("cuda")
# max_length = 512
#
# generated = model.generate(
#     input_ids=input,
#     max_length=max_length,
#     num_return_sequences=5,
#     temperature=1.0,
#     top_k=1000,
#     pad_token_id=tokenizer.pad_token_id,
#     bos_token_id=tokenizer.bos_token_id,
#     eos_token_ids=tokenizer.eos_token_id,
#     do_sample=True,
# )
#
# break_specials = [
#     SpecialTokens.BOS_TOKEN, SpecialTokens.EOS_TOKEN, SpecialTokens.DEFINITION_SEP,
#     SpecialTokens.EXAMPLE_SEP, SpecialTokens.TOPIC_SEP, SpecialTokens.POS_SEP
# ]
# break_special_ids = [tokenizer.encode(e, add_prefix_space=False)[0] for e in break_specials]
# break_special_token_map = {s: i for s, i in zip(break_specials, break_special_ids)}
#
#
# for i in range(generated.size()[0]):
#     sentence_tokens = generated[i, :].tolist()
#
#
#     accum = []
#     last_special = None
#     sep_map = {}
#     for token_id in sentence_tokens:
#         if token_id in break_special_ids:
#             if last_special is not None:
#                 sep_map[last_special] = accum
#                 accum = []
#                 last_special = token_id
#             else:
#                 last_special = token_id
#         else:
#             accum.append(token_id)
#
#     sep_map[last_special] = accum
#     accum = []
#
#     decode_sep_map = {
#         tokenizer.decode([k]): tokenizer.decode(v) for k, v in sep_map.items()
#     }
#
#     print(decode_sep_map)
#
#     # decoded = tokenizer.decode([e for e in sentence_tokens if e != tokenizer.pad_token_id])
#     print(decoded)
# """
#

tokenizer.decode(tokenizer.encode("a bc", add_prefix_space=False))

tokenizer.special_tokens_map

blacklist = set(e.title for e in pickle.load(open("data/all_words.pickle", "rb")).values())

model =  modeling.GPT2LMHeadWithWeightedLossModel.from_pretrained(
    "models/urban_dictionary_cleaned_top_def_mu02_lr_0_000005_tw40"
).to("cuda")
tw40_words = urban_dictionary_scraper.generate_words(
    tokenizer,
    model,
    blacklist=blacklist,
    num=100,
)

pickle.dump(tw1_words, open("data/labeling/tw1_words.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(tw40_words, open("data/labeling/tw40_words.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

df = pd.DataFrame(
    [
        (
            word.word,
            word.definition,
            word.example.replace(,
            "tw1" if i < len(tw1_words) else "tw2",
        )
        for i, word in enumerate(itertools.chain(
            tw1_words,
            tw40_words
        ))
    ],
    columns=("word", "definition", "example", "dataset")
)

sample = df.sample(frac=1)

sample_no_dataset = sample[:]
sample_no_dataset.to_csv("fun.csv", index=False, columns=["word", "definition", "example"])

interact()

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
# model =  AutoModelWithLMHead.from_pretrained("models/en_dictionary_parsed_lr_00005/checkpoint-50000").to("cuda")
