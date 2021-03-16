#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
from collections import defaultdict, Counter
import pandas as pd
import pickle
import re
import itertools
sys.path.append("../")
sys.path.append("../title_maker_pro")
from title_maker_pro import datasets
from collections import OrderedDict
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import copy
from word_generator import WordGenerator

dataset_path = "/mnt/evo/projects/title-maker-pro/data/urban_dictionary_words.pickle"
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

blacklist = datasets.Blacklist.load("/mnt/evo/projects/title-maker-pro/models/blacklist.pickle")
blacklist.merge(
    datasets.Blacklist.from_text_lines("/mnt/evo/projects/title-maker-pro/names.txt")
)
blacklist.merge(
    datasets.Blacklist.from_text_lines("/mnt/evo/projects/title-maker-pro/names2.txt")
)

num_defs = {k.lower(): len(d.definitions) for k, d in dataset.items()}
rows = []
seen_set = set()
for i, d in sorted(enumerate(itertools.chain.from_iterable(e.definitions for e in dataset.values())), key=lambda x: x[1].upvotes, reverse=True):
    highest_ranked_def = d.word.lower() not in seen_set
    rows.append((i, d.word, d.meaning, d.examples[0], d.upvotes, d.downvotes, d.creation_epoch, num_defs.get(d.word.lower(), 0), highest_ranked_def))
    seen_set.add(d.word.lower())

pd_dataset = pd.DataFrame(
    rows,
    columns=["idx", "word", "meaning", "example", "upvotes", "downvotes", "creation_epoch", "num_defs", "highest_rank_def"]
)

def cut(original, f, name):
    n = original[f]
    print(f"{name} cut by {100 * (1 - (len(n) / (len(original)))):.2f}% ({len(original)} -> {len(n)})")
    return n

def probably_name_meaning(meaning):
    m = re.search(r"(^|\b)(boy|girl)", meaning)
    if m:
        return m.start() < 20

    return False

t = pd_dataset.copy()
# t["upvote_percentage"] = t["upvotes"] / (t["upvotes"] + t["downvotes"] + 5)
t = cut(t, ~(t["word"].apply(blacklist.contains)), name="blacklist")
t = cut(t, (t["num_defs"] >= 2), name="min_definitions")
t = cut(t, t["highest_rank_def"], name="only_best_def")
#t = cut(t, ~(t["word"].apply(blacklist.contains)), name="blacklist")
#t = cut(t, ~(t["word"].apply(lambda x: x[:1].isupper())), name="uppercase")
t = cut(t, (t["word"].apply(lambda x: len(x.split()) <= 3)), name="max_words")
t = cut(t, (t["word"].str.len() >= 4), name="min_len")
t = cut(t, ((t["example"].str.len() + t["meaning"].str.len() + t["word"].str.len()) < 250), name="length")
t = cut(t, ~(t["meaning"].apply(probably_name_meaning)), name="name_definitions")

#t = cut(t, (t["upvote_percentage"] >= 0.5), name="upvote_percentage")

valid_indexes = set(t["idx"])
cleaned_dataset = OrderedDict()
i = 0
num_defns = 0
for k, ud_word in dataset.items():
    good_defns = []
    for d in ud_word.definitions:
        if i in valid_indexes:
            good_defns.append(copy.deepcopy(d))
            num_defns += 1
        i += 1

    if good_defns:
        new = copy.deepcopy(ud_word)
        for defn in good_defns:
            if sum(1 for c in defn.word if c.isupper()) == 1:
                defn.word = defn.word.lower()
        new.definitions = good_defns
        cleaned_dataset[k] = new

cleaned_dataset_path = "/mnt/evo/projects/title-maker-pro/data/urban_dictionary_250_top_defs.pickle"
with open(cleaned_dataset_path, "wb") as f:
    pickle.dump(cleaned_dataset, f, pickle.HIGHEST_PROTOCOL)

# nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', use)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
blacklist = datasets.Blacklist.load("/mnt/evo/projects/title-maker-pro/models/blacklist_urban_dictionary.pickle")
model = AutoModelWithLMHead.from_pretrained("/mnt/evo/projects/title-maker-pro/models/urban_dictionary_250_cleaned_top_defs_lr_00002_b9/checkpoint-50000").to("cuda:0")# model = AutoModelWithLMHead.from_pretrained("/mnt/evo/projects/title-maker-pro/models/urban_dictionary_250_cleaned_lr_00005_b9_seed4/checkpoint-140000").to("cuda:0")

words, stats = datasets.UrbanDictionaryDataset.generate_words(
    tokenizer, model,
    num=20000,
    max_iterations=1000,
    blacklist=blacklist,
    generation_args=dict(
        top_k=50,
        num_return_sequences=250,
        max_length=250,
        do_sample=True,
    ),
    dedupe_titles=True,
    filter_proper_nouns=False,
    min_definition_words=3,
)

words, stats = datasets.UrbanDictionaryDataset.generate_words(
    tokenizer, model,
    prefix=f"{datasets.SpecialTokens.BOS_TOKEN}client",
    num=100,
    max_iterations=10,
    blacklist=blacklist,
    generation_args=dict(
        top_k=50,
        num_return_sequences=250,
        max_length=250,
        do_sample=True,
    ),
    dedupe_titles=True,
    filter_proper_nouns=False,
    min_definition_words=3,
)

datasets.GeneratedWord.print_words(words)

import os
from title_maker_pro.bad_words import ULTRA_BAD_REGEX
from website.words import WordIndex, Word
from word_service.word_service_proto import wordservice_pb2

def clean_example(w, example):
    return re.sub(re.escape(w), w, example, flags=re.IGNORECASE)

def word_filter(words):
    filters = defaultdict(int)
    ret = []
    def run_over_all_text(pat, word):
        return (
            re.search(pat, word.word.strip(), flags=re.IGNORECASE)
            or re.search(pat, word.definition.strip(), flags=re.IGNORECASE)
            or re.search(pat, word.example.strip(), flags=re.IGNORECASE)
        )

    for word in words:
        if re.search(r"(^|\b)nig+", word.word.strip()):
            filters["nig"] += 1
        elif re.search(r"(^|\b)mex+", word.word.strip()):
            filters["mex"] += 1
        elif run_over_all_text(r"(\b|^)fagg+ots*", word):
            filters["fggot"] += 1import random
random.choice(list(cleaned_dataset.values()))
        elif run_over_all_text(r"(\b|^)f+a+g+", word):
            filters["fg"] += 1
        elif run_over_all_text(r"ghettos?", word):
            filters["ghetto"] += 1
        elif run_over_all_text(r"skanks*", word):
            filters["sknk"] += 1
        elif run_over_all_text(r"(^|\b)p+a+k+i+(\b|$)", word):
            filters["pki"]
        elif run_over_all_text(r"(^|\b)cunt+", word):
            filters["cnt"] += 1
        elif run_over_all_text(r"(^|\b)indian($|\b)", word):
            filters['indian'] += 1
        elif run_over_all_text(r"c+h+i+n+k+", word):
            filters['chnk'] += 1
        elif run_over_all_text(r"nigga+s*", word):
            filters['ngga'] += 1
        elif run_over_all_text(r"(^|\b)slap+s*(^|\b)", word):
            filters['slap'] += 1
        elif run_over_all_text(r"(^|\b)r+a+p+e+s*(^|\b)", word):
            filters['rape'] += 1
        elif ULTRA_BAD_REGEX.search(word.word.strip()):
            filters["ultra_bad_word"] += 1
        elif ULTRA_BAD_REGEX.search(word.definition.strip()):
            filters["ultra_bad_def"] += 1
        elif ULTRA_BAD_REGEX.search(word.example.strip()):
            filters["ultra_bad_example"] += 1
        else:
            ret.append(word)

    for k,v in sorted(filters.items()):
        print(f"Filter '{k}' removed {100 * v / len(words):.2f}%")

    print(f"Total removed {100 * (1 - len(ret) / len(words)):.2f}%")

    return ret

from hyphen import Hyphenator
h_en = Hyphenator('en_US')

wi = WordIndex(
    [
        Word(
            word=w.word,
            definition=w.definition,
            pos=w.pos,
            topic=w.topic,
            example=clean_example(w.word, w.example),
            syllables=h_en.syllables(w.word),
            probably_exists=False,
            dataset_type=wordservice_pb2.DatasetType.UD_UNFILTERED,

        ) for w in words

    ]
)
wi.dump_encrypted("../website/data/words_ud_unfiltered.enc.gz", fernet_key=os.environ.get("FERNET_ENCRYPTION_KEY"))

wg = WordGenerator(
    device="cuda:0",
    forward_model_path="/mnt/evo/projects/title-maker-pro/models/urban_dictionary_250_cleaned_lr_00005_b9_seed4/checkpoint-140000",
    inverse_model_path=None,
    blacklist_path="/mnt/evo/projects/title-maker-pro/models/blacklist.pickle",
    quantize=False,
    is_urban=True,
)

wg.generate_definition("cummy")

from word_service.word_service_proto import wordservice_pb# model = AutoModelWithLMHead.from_pretrained("/mnt/evo/projects/title-maker-pro/models/urban_dictionary_250_cleaned_lr_00005_b9_seed4/checkpoint-140000").to("cuda:0")

