import unicodedata
import re
import numpy as np
import os
import io
import string

# input is sentence, output is emotion cause pairs
# Determine clauses by splitting on punctuation.

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def remove_nonascii(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿<>\\]+", " ", w)

  w = w.strip()
  return w


def preprocess_sentence(w):
  w = remove_nonascii(w)
  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

# remove puncutation from within a cause in a doc else a cause is going to be split between different clauses.
def remove_puncutation_in_cause(text):
    # punctuation marks 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  
    start = text.find("<cause>")

    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    start_text = text[:start]
    cause = re.findall('<cause>(.*?)<\\\cause>', text)[0]
    cause = "".join(c for c in cause if c not in string.punctuation)
    end = text.find("<\cause>")
    end_text = text[end+8:] 

    text = start_text + "<cause>" + cause + "<\cause>" + end_text
    return text

def extract_cause(text):
  cur_cause=''
  try:
    cur_cause = re.findall('<cause>(.*?)<\\\cause>', text)[0]
    # Remove tags from line
    text=re.sub('<cause>', '', text)
    text=re.sub('<\\\cause>', '', text)
  except:
    pass
  return (cur_cause, text)

def has_emotion_tag(clause):
  clause = clause.strip() # remove whitespace at beg & end
  #if len(re.findall('<(.*?)>', clause)) > 0 and len(re.findall('<\\\(.*?)>', clause)) > 0 :
  if len(re.findall('<([a-z]+)>', clause)) > 0 and len(re.findall('<\\\(.*?)>', clause)) > 0 :
    return True
  return False

print(has_emotion_tag("<happy>I suppose I am happy<\happy> jjj"))
print(has_emotion_tag("I suppose I am happy<\happy>"))
print(has_emotion_tag("<happy>I suppose I am happy<\happy>asdf afe"))
print(has_emotion_tag(" <sad> I suppose I am happy<\sad> "))
print(has_emotion_tag(" <cause> I suppose I am happy<\cause> "))
print(has_emotion_tag("<happy>Indeed , the two M boys themselves were not totally happy at first<\happy>  , <cause>at having these initially undisciplined children visiting their home and generally being around on a regular basis<\cause> . "))

def extract_emotion(clause):
  clause = clause.strip()
  all_tags = re.findall('<(.*?)>', clause)
  for tag in all_tags:
    if tag!='cause' and tag!='\cause':
      return tag
  return ""
  #return re.findall('<(.*?)>', clause)[0]

print(extract_emotion("<happy>I suppose I am happy<\happy>"))
print(extract_emotion("<cause>I suppose I am happy<\cause>"))

def remove_emotion_tag_from_clause(clause, emotion):
  clause = clause.strip()
  if emotion not in clause:
    raise Exception("Emotion: ", emotion, "not present in clause: ", clause)
  
  # removing emotion tag in clause
  emotion_open_tag = '<' + emotion + '>'
  emotion_close_tag = '<\\\\' + emotion + '>'
  clause=re.sub(emotion_open_tag, '', clause)
  clause=re.sub(emotion_close_tag, '', clause)
  return clause

remove_emotion_tag_from_clause("<happy>I suppose, I am happy<\happy>", "happy")
remove_emotion_tag_from_clause("<happy>I suppose, I am happy<\happy> what's the reason", "happy")

# need to remove punctuation within a clause containing an emotion otherwise our emotion clause is divided between 2 different clauses
# warning: remove punctuation only from within the text of emotion open & close tags and NOT outside: TODO!!
def remove_punctuation_from_emotion_clause(clause):
  if not has_emotion_tag(clause):
    raise Exception('Clause has no emotion tag in it')
  punctuations = '''!()-[]{};:'"./?@#$%^&*_~,'''

  emotion = extract_emotion(clause)
  emotion_open_tag = '<' + emotion + '>'
  emotion_close_tag = '<\\' + emotion + '>'

  emotion_start = clause.find(emotion_open_tag)
  text_before_emotion_tag = clause[:emotion_start]
  print('text_before_emotion_tag: ', text_before_emotion_tag)
  emotion_end = clause.find(emotion_close_tag)
  print('emotion_end: ' , emotion_end)
  text_after_emotion_tag = clause[emotion_end+len(emotion)+3:]
  print('text_after_emotion_tag: ', text_after_emotion_tag)
  emotion_text = clause[emotion_start+len(emotion)+2:emotion_end]
  print('emotion_text: ', emotion_text)
  emotion_without_punc = "".join(c for c in emotion_text if c not in punctuations)
  
  return text_before_emotion_tag + emotion_open_tag + emotion_without_punc + emotion_close_tag + text_after_emotion_tag

  #emotion_clause_with_strings = re.findall('<cause>(.*?)<\\\cause>', text)[0]

  """
  # traverse the given string and if any punctuation 
  # marks occur replace it with null 
  start_text = text[:start]
  cause = re.findall('<cause>(.*?)<\\\cause>', text)[0]
  cause = "".join(c for c in cause if c not in string.punctuation)
  end = text.find("<\cause>")
  end_text = text[end+8:] 
  """

  #clause = "".join(c for c in clause if c not in punctuations)
  #return clause

remove_punctuation_from_emotion_clause("i am great <sad>I suppose, I am happy<\sad> , go to the bakery")

# test that extract_cause() is indeed able to extract what's within <cause>xxxxxx</cause> tags.
extract_cause("<happy>I suppose I am happy , <cause>being so, ` tiny'<\cause> ; it means I am able to surprise people with what is generally seen as my confident and outgoing personality . <\happy>")

def clean_filter_clauses(all_clauses):
  cause = ''
  clauses=[]
  for clause in all_clauses:
    e_cause, e_text = extract_cause(clause)
    if e_cause!='':
      cause = remove_nonascii(e_cause)
    if e_text!='' and remove_nonascii(e_text).strip()!='':
      clauses.append(remove_nonascii(e_text))
  if cause in ['', None]:
    raise Exception('cause is empty', clauses, all_clauses)
  return cause, clauses