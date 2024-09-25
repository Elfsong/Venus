# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2024-09-24

from datasets import load_dataset

class DataConstructor:
    def __init__(self, lang):
        self.lang = lang
        self.ds = load_dataset("Elfsong/venus_case", self.lang, download_mode="force_redownload")
        
        