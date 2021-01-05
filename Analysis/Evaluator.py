import Util as util
import Filtering as filter
import numpy as np

# this function takes a list of antonympairs (list of pairs) and a set of testwords.
# returns: Accuracy
# We assume the testwords come antonym pairs (list of lists)
def acc_ant_pair(common_data, antonym_pair, test_words):
    # project everythin down
    axis = util.get_common_axis(common_data, antonym_pair[0], antonym_pair[1])
    words=[word for pair in test_words for word in pair]
    scores = util.project_on_axis(common_data,axis, words=words)
    # reshape
    scores = np.sign(np.array(scores)).reshape(-1, 2)
    # return accuracy. If it is closer to cat0, then the return value will be >0. Else <0. Take this into account when looking for errors
    true_cat0_pred = sum(scores[:, 0] == 1)
    true_cat0_not_pred = sum(scores[:, 0] == -1)
    true_cat1_pred = sum(scores[:, 1] == -1)
    true_cat1_not_pred = sum(scores[:, 1] == 1)
    return ((true_cat0_pred + true_cat1_pred) / (len(words)))


def acc_pc_axes(common_data,axes,test_words):
    scores=[]
    words = [word for pair in test_words for word in pair]
    for axis in axes:
        proj_val=util.project_on_axis(common_data,axis, words=words)
        scores+=[proj_val]
    #make so that all projections of a single words are in one row
    scores=np.array(scores).reshape(-1,len(words))
    #take avg over axes
    scores=np.mean(scores,axis=0)
    # print('kp',list(zip(scores,words)))
    # reshape
    scores = np.sign(scores).reshape(-1, 2)
    # return accuracy. If it is closer to cat0, then the return value will be >0. Else <0. Take this into account when looking for errors
    true_cat0_pred=sum(scores[:,0]==1)
    true_cat0_not_pred=sum(scores[:,0]==-1)
    true_cat1_pred=sum(scores[:,1]==-1)
    true_cat1_not_pred=sum(scores[:,1]==1)
    return ((true_cat0_pred + true_cat1_pred) / (len(words)))


def confusion_mat(common_data,axes,test_words):
    scores = []
    words = [word for pair in test_words for word in pair]
    for axis in axes:
        proj_val = util.project_on_axis(common_data, axis, words=words)
        scores += [proj_val]
    # make so that all projections of a single words are in one row
    scores = np.array(scores).reshape(-1, len(words))
    # take avg over axes
    scores = np.mean(scores, axis=0)
    # reshape
    scores = np.sign(scores).reshape(-1, 2)
    true_cat0_pred=sum(scores[:,0]==1)
    true_cat0_not_pred=sum(scores[:,0]==-1)
    true_cat1_pred=sum(scores[:,1]==-1)
    true_cat1_not_pred=sum(scores[:,1]==1)
    return np.array([true_cat0_pred,true_cat0_not_pred,true_cat1_not_pred,true_cat1_pred]).reshape((2,2))



def calc_skew_score(common_data,axis,professions_with_scores):
    #calc the cosine sim:
    all_proj=util.project_all_on_axis(common_data,axis)
    filtered=filter.filter_words(common_data,[word for word,d in professions_with_scores], typ='professions')
    #def_skew=np.array([util.skew_score(all_proj[word],d,s) for word,d,s in professions_with_scores if word in filtered])
    #defin = np.array( [d for word, d, s in professions_with_scores if word in filtered])
    #stereo= np.array( [s for word, d, s in professions_with_scores if word in filtered])
    #skew=np.array([util.skew_score(all_proj[word],d,s) for word,d,s in professions_with_scores if word in filtered])
    # def_skew_avg=np.mean(def_skew)
    #skew_avg=np.mean(skew)
    #SUPER SIMPLE
    skew=[all_proj[word] for word in filtered]
    #def_skew_std=np.std(def_skew)
    #stereo_skew_std=np.std(stereo_skew)
    #return (def_skew_avg,def_skew_std,stereo_skew_avg,stereo_skew_std)
    return skew,filtered