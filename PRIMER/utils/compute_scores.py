import py_vncorenlp
# Download the word and sentence segmentation component
# py_vncorenlp.download_model(save_dir='/home/redboxsa_ml/sonlh/data-env')
# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/redboxsa_ml/sonlh/data-env')
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
from transformers import AutoTokenizer

def compute_scores(cluster, scorer):
    
    """
    input:
        cluster: list of list of str, each str is sentence.
        scorer: rouge scorer
    output:
        [[{'text': 'I like apple',
            'pegasus_score': 0.3,
            'pyramid_rouge': 0.5},
          {'text': 'And I also like banana',
            'pegasus_score': 0.5,
            'pyramid_rouge': 0.4}],
          ...],
          [{'text': ...,
            'pegasus_score': 0.4,
            'pyramid_rouge': 0.6},
          ...],
        ...]
    """
    # cluster = [[s for p in doc.split('\n') for s in sent_tokenize(p) if s!=''] for doc in documents.split('|||||')]
    result_dict = []
    all_text = "\n".join(["\n".join(d) for d in cluster])
    for i_doc, doc in enumerate(cluster):
        result_dict.append([])
        
        for i_s, s in enumerate(doc):
            output=rdrsegmenter.word_segment(s)
            segmented_text= ''.join(output)  
            # if s is too short, i.e. less than 5 chars, we directly set scores to 0
            if len(s.split()) < 5:
                result_dict[-1].append(
                    {"text": segmented_text, "pegasus_score": 0, "pyramid_rouge": 0}
                )
                continue
            # compute pegasus_score
            # ref_sents = all_text.replace(s, "")
            # score = compute_rouge_scores(scorer, [s], [ref_sents])
            # pegasus_score = score["rouge1"][0].fmeasure
            pegasus_score=0

            # compute pyramid rouge score (Cluster ROUGE in the paper)
            pyramid_score = 0
            for i_doc_pyramid, ref in enumerate(cluster):
                if i_doc_pyramid != i_doc:

                    # whole doc version, the rouge scores are computed based on
                    # ROUGE(s_n,doc_m)
                    hyp = [s]
                    ref = [" ".join(ref)]
                    scores = compute_rouge_scores(scorer, hyp, ref)
                    pyramid_score += (
                        scores["rouge1"][0].fmeasure
                        + scores["rouge2"][0].fmeasure
                        + scores["rougeL"][0].fmeasure
                    ) / 3

                else:
                    continue
            result_dict[-1].append(
                {
                    "text": segmented_text,
                    # "segment_text":segmented_text,
                    "pegasus_score": pegasus_score,
                    "pyramid_rouge": pyramid_score,
                }
            )
    return result_dict


def compute_rouge_scores(scorer, predictions, references):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]

    assert len(predictions) == len(references)
    all_scores = []
    for pred, ref in zip(predictions, references):
        all_scores.append(scorer.score(target=ref, prediction=pred))
    final_scores = {}
    for score_type in all_scores[0].keys():
        final_scores[score_type] = [
            all_scores[i][score_type] for i in range(len(all_scores))
        ]
    return final_scores


# def get_entities(nlp, all_docs):
#     all_entities_pyramid = {}
#     all_entities = {}
#     for i, doc in enumerate(all_docs):
#         all_entities_cur = set()
#         for s in doc:
#             sent = nlp(s["text"])
#             if len(sent.ents) != 0:
#                 for ent in sent.ents:
#                     all_entities_cur.add(ent.text)
#                     all_entities[ent.text] = all_entities.get(ent.text, 0) + 1
#         for e in all_entities_cur:
#             all_entities_pyramid[e] = all_entities_pyramid.get(e, 0) + 1
#     return all_entities_pyramid, all_entities

# def get_entities(nlp, all_docs):
#   all_entities_pyramid = {}
#   all_entities = {}
#   for i, doc in enumerate(all_docs):
#       all_entities_cur = set()
#       for s in doc:
#           sent = nlp.annotate(s["text"])
#           ent_list = out_phoNLP(sent)
#           if len(ent_list) != 0:
#               for ent in ent_list:
#                   all_entities_cur.add(ent)
#                   all_entities[ent] = all_entities.get(ent, 0) + 1
#       for e in all_entities_cur:
#           all_entities_pyramid[e] = all_entities_pyramid.get(e, 0) + 1
#   return all_entities_pyramid, all_entities

added_tokens = ["<doc-sep>"]
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large",additional_special_tokens=added_tokens)
def get_entities(nlp, all_docs):
    all_entities_pyramid = {}
    all_entities = {}
    f=open('/home/redboxsa_ml/sonlh/news-corpus-datasets/news_corpus_sample_train/annotate.txt','a')
    for i, doc in enumerate(all_docs):
        ent_list=[]
        all_entities_cur = set()
        for s in doc:
            try:
                sent = nlp.annotate(s["text"])
                ent_list = out_phoNLP(sent)
                if len(ent_list) != 0:
                    for ent in ent_list:
                        all_entities_cur.add(ent)
                        all_entities[ent] = all_entities.get(ent, 0) + 1
            except Exception:
                print("-----------------------------------------\n\n\n\n\n\n\n\n\n\n")
                f.write(str(len(tokenizer(s["text"])['input_ids']))+ s["text"]+ '\n')
        for e in all_entities_cur:
            all_entities_pyramid[e] = all_entities_pyramid.get(e, 0) + 1
    # f.close()
    return all_entities_pyramid, all_entities

def out_phoNLP(phonlp_annot):
  tokens = phonlp_annot[0][0]
  ners = phonlp_annot[2][0]
  pos_tags = []
  for i in range(len(tokens)):
    pos_tags.append(phonlp_annot[1][0][i][0])

  conlltags = [(token, pos, ner) for token, pos, ner in zip(tokens, pos_tags, ners)]
  ne_tree = conlltags2tree(conlltags)

  original_text, entity = [], []
  for subtree in ne_tree:
      # skipping 'O' tags
      if type(subtree) == Tree:
          original_label = subtree.label()
          original_string = " ".join([token for token, pos in subtree.leaves()])
          entity.append(original_string)
          original_text.append((original_string, original_label))
  
  return entity
