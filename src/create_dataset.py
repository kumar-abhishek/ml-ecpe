from clean_text import * 

# 1. Remove any accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [document, emotion, cause, clauses list]
document=[]
cause=[]
clause=[]
clause_global=[]
emotion_label=[]
cause_label=[]
clauseid_to_docid=[]
known_emotion_cause_pair_per_doc_id=[]

def create_dataset(lines, num_examples):
  document.clear()
  cause.clear()
  clause.clear()
  clause_global.clear()
  emotion_label.clear()
  cause_label.clear()
  clauseid_to_docid.clear()
  #lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  for i, line in enumerate(lines[:num_examples]):
    line = remove_puncutation_in_cause(line)
    if has_emotion_tag(line):
      print('before remove punc', line)
      line = remove_punctuation_from_emotion_clause(line)
      print('after remove punc', line)

    # Determine clauses by splitting on punctuation.
    all_clauses = re.split("[,!;:\"]+", line) # removing dot from this for now
    print('before: ', all_clauses)
    filter_cause, filter_clauses = clean_filter_clauses(all_clauses)
    print('after: ', filter_clauses)
    cause.append(filter_cause)

    clause.append([filter_clauses])
    doc = extract_cause(line)[1]
    
    # clean up document
    clean_doc = preprocess_sentence(doc)
    document.append(clean_doc)

    clause_global.extend(filter_clauses)
    cause_found=False
    if not filter_clauses:
      raise Exception('filter_clauses empty')
    
    sub_emotion_clauses_per_doc=[]
    cur_emotion = ''

    for idx, fclause in enumerate(filter_clauses):
      clauseid_to_docid.append(i)
      fclause = fclause.strip('.')
      
      if fclause.find(filter_cause) != -1:
        if cause_found:
          pass
        else:
          cause_label.append(1)
          cause_id = len(clauseid_to_docid)-1
          cause_found=True
      else:
        cause_label.append(0)

      if fclause=='':
        continue
      print('fclause: ', fclause)

      # emotion label extract
      if has_emotion_tag(fclause):
        cur_emotion = extract_emotion(fclause)
        fclause = remove_emotion_tag_from_clause(fclause, cur_emotion)
        sub_emotion_clauses_per_doc.append(cur_emotion)
      else:
        sub_emotion_clauses_per_doc.append('')

    emotion_label.extend(sub_emotion_clauses_per_doc)
    known_emotion_cause_pair_per_doc_id.append([cur_emotion, cause_id])

    if cause_found==False:
      raise Exception('cause not found for line: ', line, i, 'filter_cause: ', filter_cause)    


  return [document, cause, clause, clause_global, cause_label, emotion_label]
