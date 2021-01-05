import Filtering as filter
from gensim.models import Word2Vec, KeyedVectors

#This class holds the actual word embeddings and common_we, which are the word embeddings reduced to common vocab
class Data_Holder:
    def __init__(self,clean=True):
        self.we = dict()
        self.we['gab'] = self.load_we('../App/src/server/data/Gab_all/Gab_all_300_sg_bigram_iter8.model')
        self.we['onebill'] = self.load_we('../App/src/server/data/OneBillion/OneBillion_300_sg_bigram_iter8.model')
        self.we['redditall'] = self.load_we(
            '../App/src/server/data/Reddit_all/Reddit_all_300_sg_bigram_iter20.model')
        print('Setting a common vocabulary...')
        self.common_we,self.common_vocab=self.set_common_vocab()
        print('Done!')



    def load_we(self, link):
        if (link[-3:] == 'bin'):
            if 'google' in link:
               return KeyedVectors.load_word2vec_format(link, binary=True)
            else:
                return KeyedVectors.load_word2vec_format(link, binary=binary)
        else:
            return Word2Vec.load(link)

    def set_common_vocab(self,clean=True):
        #first, find a common vocab
        common_vocab=[]
        for ind,dataset in enumerate(self.we.values()):
            if ind==0:
                common_vocab=set(dataset.wv.vocab)
            else:
                common_vocab=common_vocab &set(dataset.wv.vocab)
        #then, clean the vocab
        if clean:
            common_vocab=filter.filter_vocab(common_vocab)
        #get all the data for common vocab
        commons=dict()
        for name,we in self.we.items():
            commons[name]=dict(zip(common_vocab,we[common_vocab]))
        return commons,common_vocab
