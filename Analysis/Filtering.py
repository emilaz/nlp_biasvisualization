import numpy as np
import re

#filters a given vocab
def filter_vocab(vocab):
    new_vocab = []
    for word in vocab:
        if not (re.compile(r'.*[A-Z\d.,\/#!$%\^&\*;:{}=\-`~())].*.').match(word) or len(word) > 20):
            new_vocab.append(word)
    return new_vocab


#input: list of pairs of words (antonyms), common vocab
#output: list of pairs of words (antonyms), where both antonyms are in vocab
def filter_pair_words(common_vocab, unclean_words, typ):
    #print('Filtering out %s that are not in the commonly shared vocabulary...' %(typ))
    test_words = [pair for pair in unclean_words if
                  (pair[0] in common_vocab and pair[1] in common_vocab)]
    #print('Done! %d of %d pairs remaining' % (len(test_words), len(unclean_words)))
    return list(set(test_words))

def filter_words(common_vocab, unclean_words, typ):
    #print('Filtering out %s that are not in the commonly shared vocabulary...' %(typ))
    test_words = [word for word in unclean_words if word in common_vocab]
    #print('Done! %d of %d pairs remaining' % (len(test_words), len(unclean_words)))
    return test_words


#returns list of antonyms to be deleted, per dataset
def filter_antonyms(datasets,analysis_type, return_pair_no):

    if analysis_type == 'best_overall':
        # we need to retrieve the key-value pairs, because dicts are not ordered, but we are getting indices of highest values.
        items = list([list(dataset.scores.items()) for dataset in datasets])
        # here, we assume all datasets have the same amount of scores
        all_scores = np.array([item[1] for dat in items for item in dat]).reshape(-1, len(items[0]))
        all_items = [item[0] for dat in items for item in dat]
        # get all scores, then take average
        avg_scores=np.mean(all_scores,axis=0)
        #this returns all but five best scores
        bad_indices=np.argpartition(-avg_scores,return_pair_no)[return_pair_no:]
        #we delete right here, because dicts have no order
        for ind,dataset in enumerate(datasets):
            for bad in bad_indices:
                del_idx=ind*len(items[0])+bad
                del dataset.scores[tuple(all_items[del_idx])]

    if analysis_type == 'best_each':
        for ind,dataset in enumerate(datasets):
            items=list(dataset.scores.items())
            scores=np.array([item[1] for item in items])
            items=[item[0] for item in items]
            bad_indices=np.argpartition(-scores,return_pair_no)[return_pair_no:]
            for bad in bad_indices:
                del dataset.scores[items[bad]]
