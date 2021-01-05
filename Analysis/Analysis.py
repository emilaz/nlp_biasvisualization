from Common_Worker import Common_Worker as cw
import numpy as np
import Visualizations as vis
datasets=cw([('lsc','bigram'),('td','bigram')])
#datasets=cw([('google',)],False)


#read in antonym pairs and test points...
f=open('./Data/Words/antonym_pairs.txt')
antonyms = [tuple(line.rstrip('\n').split(' ')) for line in f]
antonyms=[('he','she'), ('man','woman')]
#read in test words...
f=open('./Data/Words/test_words.txt')
#test_words = [line.rstrip('\n').split(' ') for line in f]
test_words = [tuple(line.rstrip('\n').split(' ')) for line in f]


datasets.set_antonym_axes(antonyms,test_words,'best_overall')

datasets.set_bias_axis()

datasets.set_projections()

int_points=datasets.get_points_by_var('overall')

#get ten best:
#first, get list, because dict has no order
items = list(int_points.items())
all_scores = np.array([item[1] for item in items])
all_words = np.array([item[0] for item in items])
argmax=np.argpartition(-all_scores,20)[:20]
int_words=all_words[argmax]
#get the projections
projections=datasets.get_projections()
#just for now, to see.
vis.vis_variances(projections,int_words)