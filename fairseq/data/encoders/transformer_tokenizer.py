# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel, BertTokenizer
from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass
from typing import Optional

from fairseq.data.encoders import register_bpe


def get_sentence(tokens):
    sentence = []
    length = len(tokens)
    i=0
    while(i < length):
        index = [i]
        index = find_pound_key(tokens, i, index)
        i = index[-1]+1
        word = [tokens[j].strip("##") for j in index]
        word = "".join(word)
        sentence.append(word)
    return sentence

def find_pound_key(tokens, i, index):
    if i == len(tokens)-1:
        return index
    if "##" not in tokens[i+1]:
        return index
    if "##" in tokens[i+1]:
        index.append(i+1)
        return find_pound_key(tokens, i+1, index)

@dataclass
class TransformerTokenizerConfig(FairseqDataclass):
    pretrained_vocab: Optional[str] = field(
        default=None, metadata={"help": "pre-trained bpe model"}
    )

@register_tokenizer("transformer_tokenizer", dataclass=TransformerTokenizerConfig)
class TransformerTokenizer(object):
    def __init__(self, cfg):
        print(cfg.pretrained_vocab)
        if cfg.pretrained_vocab is not None:
            self.tokenizer = BertTokenizer.from_pretrained(cfg.pretrained_vocab)
        else:
            self.tokenizer = BertTokenizer.from_pretrained("./download_prepare/12k-vocab-models")


    def encode(self, x: str) -> str:
        toks = self.tokenizer.tokenize(x)
        return " ".join(toks)

    def decode(self, x: str) -> str:
        x = get_sentence(x.split(" "))
        return " ".join(x)
