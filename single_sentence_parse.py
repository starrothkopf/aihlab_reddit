import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_lg")

sentence = (
    "Chatgpt is almost unusable nowadays. It can't understand anything and it doesn't understand more from follow-up discussion. It also refuses to do the most basic of things randomly. It's completely nerfed and trashed. I almost exclusively use Gemini now"
)

doc = nlp(sentence)

displacy.serve(doc, auto_select_port=True, style='dep')


