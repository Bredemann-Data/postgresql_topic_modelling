#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:12:33 2023

@author: basti
"""

#%% function for language detection

from langdetect import detect

def lang_rec(text):

    try:
        lang = detect(text)
    except: 
        return "unknown language"
    else:
        return lang

def add_lang_column(df, spalte):
    language = df[spalte].map(lang_rec)
    #pickle.dump(language, open(path + '.p', "wb"))
    df['language'] = language
    print("languages detected")
    return df

