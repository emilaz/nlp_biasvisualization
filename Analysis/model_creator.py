import json
from nltk import word_tokenize
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
from gensim.models import Phrases
from gensim.models.phrases import Phraser

##change/add to this function as you need
# once you have the corpus tokenized, run the second/third cell to get and save your word2vec model.

with open('./Data/Reddit/RC_2018-02') as f:
    td_sentences = []
    lsc_sentences = []
    pol_sentences = []
    for line in tqdm(f):
        jobj = json.loads(line)
        if jobj['subreddit'] == 'The_Donald':
            td_sentences += [jobj['body']]
        if jobj['subreddit'] == 'LateStageCapitalism':
            lsc_sentences += [jobj['body']]
        if jobj['subreddit'] == 'politics':
            pol_sentences += [jobj['body']]

td_tokens = [word_tokenize(sent) for sent in td_sentences]
lsc_tokens = [word_tokenize(sent) for sent in lsc_sentences]
pol_tokens = [word_tokenize(sent) for sent in pol_sentences]

##this is to include bigrams
td_bi=Phraser(Phrases(td_tokens))
lsc_bi=Phraser(Phrases(lsc_tokens))
pol_bi=Phraser(Phrases(pol_tokens))

#choose sg=1 for skip-gram model WE MIGHT WANT TO CHANGE THIS
#also, more iterations might be necessary
td_model = Word2Vec(td_bi[td_tokens], size=150, min_count=5,sg=1,workers=8, iter=20)
lsc_model = Word2Vec(lsc_bi[lsc_tokens], size=150, min_count=5,sg=1,workers=8, iter=20)
pol_model = Word2Vec(pol_bi[pol_tokens], size=150, min_count=5,sg=1,workers=8,iter=20)

td_model.wv.save_word2vec_format('./Data/Reddit/td_model_feb_18_bigram.bin')
lsc_model.wv.save_word2vec_format('./Data/Reddit/lsc_model_feb_18_bigram.bin')
pol_model.wv.save_word2vec_format('./Data/Reddit/pol_model_feb_18_bigram.bin')