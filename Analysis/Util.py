import numpy as np
import warnings
import itertools

# several util functions
def cosine_sim(model, word1, word2):
    # first, get vectors of words.
    # You can pass either the word or its vector directly.
    if (isinstance(word1, str)):
        vec1 = model[word1]
    else:
        vec1 = word1
    if (isinstance(word2, str)):
        vec2 = model[word2]
    else:
        vec2 = word2
    # first, normalize vectors:
    vec1_normed = vec1 / np.linalg.norm(vec1)
    vec2_normed = vec2 / np.linalg.norm(vec2)
    return vec1_normed @ vec2_normed.T


def get_common_axis(model, bias_cat1, bias_cat2):
    #normalize vectors before adding
    axis = model[bias_cat1] / np.linalg.norm(model[bias_cat1]) - model[bias_cat2] / np.linalg.norm(
        model[bias_cat2])
    # normalize
    axis = axis / np.linalg.norm(axis)
    return axis

# returns the cosine similarities for a given set of words (or the whole vocab) for a given antonym pair
# return POS value, if more similar to cat1, else NEG value
#if antonym_pair=None, we use the bias axis
def project_on_axis(dataset,axis,words):
    scores = []
    for word in words:
        sim=cosine_sim(dataset, axis, word)
        scores += [sim]
    scores=np.array(scores)
    return scores

def project_all_on_axis(dataset,axis,normalize=True):
    scores = dict()
    # if no specific range of words are given, choose all.
    words = dataset.keys()
    for word in words:
        # we need to pass the model here because we might want to project using diff datasets (common, orig...)
        scores[word] = cosine_sim(dataset, axis, word)
    # normalize the scores????
    if normalize:
        normalizer = abs(scores[max(scores, key=lambda y: abs(scores[y]))])
        for key in scores.keys():
            scores[key] = scores[key] / normalizer
    return scores

def n_closest_words(model, target_word, n):
    sim = [(cosine_sim(target_word, word), word) for word in model.wv.vocab]
    sim_sorted = sorted(sim, key=lambda x: x[0], reverse=True)
    return sim_sorted[:n]

#the higher the score, the more skewed
def skew_score(projection,definit):
    #score=abs(projection-definit)**2/abs(projection-stereo)
    #score=stereo-(projection-definit)
    #score = (projection - definit)
    score =  projection
    return score

def calc_normalized_proj(common_data,axis,professions):
    all_proj = project_all_on_axis(common_data,axis)
    filtered = filter.filter_words(common_data,[word for word in professions], typ='professions')
    projs = [all_proj[word] for word in filtered]
    return projs


#Takes two projected data sets (dataset.projected_data) and finds words that differ most between them
def get_greates_vec_diffs(proj1,proj2):
    diffs=[]
    #both should have the same keys at this point (common vocab)
    for word in proj1.keys():
        #find difference in word_sim
        diffs+=[(word,abs(proj1[word]-proj2[word]))]
    word_diffs=sorted(diffs, key=lambda x:x[1], reverse=True)
    return word_diffs

#if method==overall, this returns a dict of words (and values?) that have the greates variance across datasets
#if method==pairwise, this returns a dict of pairwise embeddings as keys and a word:var dict as value.
def greatest_variance(projections,method='pairwise'):
    if method not in ['pairwise','overall']:
        raise ValueError("Given method must be 'pairwise' or 'overall'")
    if method=='pairwise':
        warnings.warn('The given method takes an exponentially long time given the input. Make sure the amount of datasets is not too big')
        iters=itertools.combinations(projections.keys(),2)
        pairwise_vars=dict()
        for iter in iters:
            pairwise_vars[iter]=get_greates_vec_diffs(projections[iter[0]],projections[iter[1]])
        return pairwise_vars
    if method=='overall':
        overall_vars=dict()
        #apply sample variance
        #first, get mean per projected vector (we do this complicated thing to get the words)
        for word in projections[list(projections.keys())[0]].keys():
           word_mean=np.mean([proj[word] for proj in projections.values()])
           overall_vars[word]=np.mean(np.array([proj[word]-word_mean for proj in projections.values()])**2)
        return overall_vars
    return None





