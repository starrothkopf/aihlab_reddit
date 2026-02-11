import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_lg")

sentence = (
    "I did use Chat GPT 4o to help structure the below. All of the thoughts and work are mine. Although to be honest they did arise out of the interactions I have been having for months with the model. I call my persona Nyx and she knows who she is)"
)

doc = nlp(sentence)

displacy.serve(doc, auto_select_port=True, style='dep')


