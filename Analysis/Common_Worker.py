import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import Util
import Evaluator as eval
import Filtering as filter
from DatasetProvider import Dataset



#this worker sets values for all datasets in synch. It should not hold important data itself.
class Common_Worker:
    #current thing
    def __init__(self, common_we,common_vocab):
        #initialize all datasets.
        #we save them in a dict, with a combination of the given keywords as key.
        self.all_datasets = dict()
        for name,common_words_we in common_we.items():
            self.all_datasets[name] = Dataset(common_words_we)
        #set a common vocabulary for all
        self.common_vocab = common_vocab


    def setup(self, unclean_antonyms, unclean_test_words,anal_type, n_comps=1):
        if anal_type not in ['all','best_overall','best_each', 'cv']:
            raise ValueError("Analysis Type must be 'all','best_overall','cv', or 'best_each'.")
        #clean words
        test_words = filter.filter_pair_words(self.common_vocab, unclean_test_words, 'test words')
        antonyms = filter.filter_pair_words(self.common_vocab, unclean_antonyms, 'antonyms')
        if anal_type == 'cv':
            self.set_with_cv(antonyms,n_comps)
        else:
            #calc all scores
            self.set_antonym_scores(antonyms,test_words)
            #filter out bad pairs based on method
            filter.filter_antonyms(self.all_datasets.values(), anal_type, min(5, len(antonyms) - 1))
            #get the axes
            self.set_antonym_axes()
            self.set_bias_axis(n_comps)
        #returns clean test_words
        return test_words

    def set_with_cv(self,antonyms,n_comps):
        for name,dataset in self.all_datasets.items():
                dataset.principal_dir = dataset.pca_cv(antonyms,n_comps)

    #this function calculates antonym scores for given set of antonyms 
    #returns a list of antonyms that scored below a given threshold
    def set_antonym_scores(self, antonyms, test_words):
        for name,dataset in self.all_datasets.items():
            for p in antonyms:
                dataset.scores[p] = eval.acc_ant_pair(dataset.common_data,antonym_pair = p,test_words = test_words)
            #print(dataset.scores)

    #this functions gets the antonym scores using get_antonym_scores
    #on the remaining antonyms (those above threshold), and sets their axes for each dataset.
    def set_antonym_axes(self):
        for dataset in self.all_datasets.values():
            # print('das sind die scores:', dataset.scores)
            axes = [Util.get_common_axis(dataset.common_data,b[0],b[1]) for b in dataset.scores.keys()]
            ###new
            new_axes = [-ax for ax in axes]
            axes+=new_axes
            ####
            dataset.new_axes=axes

    def set_bias_axis(self, n_comps):
        for name,dataset in self.all_datasets.items():
            dataset.set_bias_axis(name, n_comps)


    #Input: a list of dataset names (must be same as in json file)
    #Output:dictionary with chosen antonyms per dataset
    def get_chosen_antonyms(self,dataset_names = None):
        ret = dict()
        if dataset_names is None:
            for name,data in self.all_datasets.items():
                ret[name] = data.scores.keys()
        else:
            for name in dataset_names:
                ret[name] = self.all_datasets[name].scores.keys()
        return ret

    #returns a dict with key = dataset name, value =  its projections (dict of word, value)
    def get_projections(self):
        ret = dict()
        for name,data in self.all_datasets.items():
            ret[name] = data.projections
        return ret


    def get_points_by_var(self,method = 'pairwise'):
        projs = self.get_projections()
        vars = Util.greatest_variance(projs,method)
        return vars


    def get_skew_scores(self, profesions):
        skews = dict()
        for name,data in self.all_datasets.items():
            skews[name],filtered = eval.calc_skew_score(data.common_data,data.principal_dir,profesions)
        return skews,filtered

    def get_confusion_mats(self,test_words):
        all_mat = dict()
        for name,data in self.all_datasets.items():
                all_mat[name] = eval.confusion_mat(data.common_data,data.principal_dir,test_words)
        return all_mat


    def get_pc_acc(self,test_words,axes=None):
        all_mat  =  dict()
        for name, data in self.all_datasets.items():
            if axes is None:
                all_mat[name] = eval.acc_pc_axes(data.common_data, data.principal_dir, test_words)
            else:
                all_mat[name] = eval.acc_pc_axes(data.common_data, axes, test_words)
        return all_mat

# ##other interesting functions that might be worth checking out later....

# # #check out the closes words to 
# # word = 'Jew'
# # #get 100 closest
# # closest_pol = n_closest_words(model_pol,word, 10)

# # #check out the closes words to 
# # word = 'Jew'
# # #get 100 closest
# # closest_pol = n_closest_words(model,word, 100)


# # vis_pc(model,closest)

# # #now, do on axis
# # m = 'Muslim'
# # c = 'Christian'
# # j = 'Jew'
# # mc_scores = []
# # mj_scores = []
# # cj_scores = []
# # for word in closest:
# #     mc_scores+ = [cosine_sim(model,m, word)-cosine_sim(model,c, word)]
# #     mj_scores+ = [cosine_sim(model,m, word)-cosine_sim(model,j, word)]
# #     cj_scores+ = [cosine_sim(model,c, word)-cosine_sim(model,j, word)]

# # vis_along_axis(mc_scores,mj_scores,closest, 'Christian-Muslim','Jew-Muslim')
# # vis_along_axis(cj_scores,mj_scores,closest, 'Jew-Christian','Jew-Muslim')
    

# #common_google,common_gab = get_common_voc_vec(model_google,model_gab)
# common_google,common_pol = get_common_voc_vec(model_google,model_pol)

# bc1 = 'he'
# bc2 = 'she'
# projected_pol = project_on_axis(common_pol,bc1, bc2,normalize = True)
# projected_google = project_on_axis(common_google,bc1, bc2,normalize = True)

# projected_pol['He']

# #find the greates difference in similarities
# google_vs_pol = get_greates_vec_diffs(projected_pol,projected_google)

# print(google_vs_pol[:10])

# top_100_diffs = google_vs_pol[0:90]

# ##THIS SAVES THE FIGURE! DO YOU WANT THIS?
# fig.savefig(title)

# ##sanity check:
# # print(cosine_sim(twitt_common,'jew','islamic'))
# # print(cosine_sim(model_twitt,'jew','islamic'))
# # print(cosine_sim(twitt_common,'muslim','islamic'))
# # print(cosine_sim(model_twitt,'muslim','islamic'))
# print(cosine_sim(common_pol,'he','he'))
# print(cosine_sim(common_pol,'he','she'))
# print(projected_pol['he'])

# #take the axis from the paper:

# #as sanity check, see where these words end up
# #heshe_axis_google = common_google['he']-common_google['she']
# #google_score = [cosine_sim(common_google,heshe_axis_google,word) for pair in all_pairs for word in pair]
# #heshe_axis_gab = common_gab['he']-common_gab['she']
# #gab_score = [cosine_sim(common_gab,heshe_axis_gab,word) for pair in all_pairs for word in pair]

# print('Where do the words of each female-male pair end up on the she-he axis?')
# print(np.array(google_score).reshape(-1,2))
# print(np.array(gab_score).reshape(-1,2))


# # gender_specific_score = np.array([projected_google.get(word,2) for word in gender_specific])

# #filter for words that are not in the common vocab
# test_pairs = [pair for pair in test_pairs if (pair[0] in common_pol.keys() and pair[1] in common_pol.keys())]


# In[ ]:


# gender_specific_score = np.array([projected_google.get(word,2) for word in gender_specific])
# gender_specific_score<2
# gender_spec_in_vocab = gender_specific[gender_specific_score<2]
# gender_spec_in_vocab_score = np.sign(gender_specific_score[gender_specific_score<2])
# combined_list = [list(a) for a in zip(gender_spec_in_vocab,gender_spec_in_vocab_score)]
#
#
# # In[ ]:
#
#
# scores_google = [get_antonym_scores(antonym_pair = p,model = common_google,test_words = [word for pair in test_pairs for word in pair]) for p in antonym_pairs]
# #scores_gab = [get_antonym_scores(antonym_pair = p,model = common_gab,test_words = [word for pair in test_pairs for word in pair]) for p in antonym_pairs]
# scores_pol = [get_antonym_scores(antonym_pair = p,model = common_pol,test_words = [word for pair in test_pairs for word in pair]) for p in antonym_pairs]

