import Util as util
import Filtering as filter
import Visualizations as vis
import Evaluator as eval
from sklearn.decomposition import PCA
from random import shuffle
import itertools



class Dataset:

    def __init__(self,common_data):
        # these are variables to be filled up by a separate merger class
        self.common_data = common_data
        self.new_axes = None
        self.principal_dir=None
        self.projections=None
        self.scores=dict()



    def set_bias_axis(self,name, n_comps):
        #this return the min amount of PD we need to explain the given amount of var in the data
        #vis.elbow_curve(self.new_axes,name)
        #run pca on all antonym pairs, desiring that many axes
        pca=PCA(n_components=n_comps)
        pca.fit(self.new_axes)
        #check how they score better:
        #for
        ax=self.best_pc_combo(pca.components_)
        self.principal_dir=ax

    def best_pc_combo(self,axes):
        #we test on words used for calculating PC axis
        #for now, hardcode for max 2 axes
        if len(axes)==1:
            axes_with_neg = [axes, -axes]
            best=0
            axis=None
            words=self.scores.keys()
            for ax in axes_with_neg:
                sc=eval.acc_pc_axes(self.common_data,ax,words)
                if sc>best:
                    best=sc
                    axis=ax
        else:
            axes_with_neg_first = [axes[0], -axes[0]]
            axes_with_neg_second = [axes[1], -axes[1]]
            combos=list(itertools.product(axes_with_neg_first, axes_with_neg_second))
            best = 0
            axis = None
            for ax in combos:
                sc = eval.acc_pc_axes(self.common_data, ax, self.scores.keys())
                if sc > best:
                    best = sc
                    axis = ax
        return axis

    def pca_cv(self,antonyms, n_comps):
        #shuffle the words
        ants=antonyms.copy()
        shuffle(ants)
        best=None
        best_score=0
        #do PCA on 5, test on all
        for i in range(0,len(antonyms),5):
            if(i==len(antonyms)-1):
                continue
            pca=PCA(n_components=n_comps)
            axes=[util.get_common_axis(self.common_data,a[0],a[1]) for a in ants[i:i+5]]
            ##new thing:
            new_axes=[-ax for ax in axes]
            axes+=new_axes
            ###
            pca.fit(axes)
            acc=eval.acc_pc_axes(self.common_data, pca.components_,antonyms)
            if acc>best_score:
                best_score=acc
                best=pca.components_.copy()
        return best




    #def calc_bias_score(self):

