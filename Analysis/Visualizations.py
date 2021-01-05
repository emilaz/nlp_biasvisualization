import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def elbow_curve(new_axes, dataset_name=None):
    components = range(1, new_axes.shape[0] + 1)
    explained_variance = []
    # till where?
    lim = min(100, new_axes.shape[0])
    count = 0
    for component in components[:lim]:
        pca = PCA(n_components=component)
        pca.fit(new_axes)
        expl_var = sum(pca.explained_variance_ratio_)
        explained_variance.append(expl_var)
        count += 1
    plot = plt.scatter(
        x=components[:count], y=explained_variance)
    plt.xlabel('Number of Principal Directions')
    plt.ylabel('Explained Variance')
    if dataset_name:
        plt.title('Elbow curve for dataset %s' % dataset_name)
    plt.show()

# vis functions
def vis_pc(dataset, words):
    X = dataset[words]
    X.shape
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # words = list(dataset.wv.vocab)
    # get limits
    ylim = np.max(np.abs(result[:, 1]))
    xlim = np.max(np.abs(result[:, 0]))
    plt.figure(figsize=(8, 8))
    plt.ylim([-ylim, ylim])
    plt.xlim([-xlim, xlim])
    plt.scatter(result[:, 0], result[:, 1])
    plt.plot([0] * 100, np.linspace(-ylim, ylim, 100), 'k--', linewidth=1)
    plt.plot(np.linspace(-xlim, xlim, 100), [0] * 100, 'k--', linewidth=1)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Visualization of PCA')
    # for i, word in enumerate(words[:100]):
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

def vis_two_axis(sims1, sims2, words, axis1, axis2):
    ylim = max(abs(sims2))
    xlim = max(abs(sims1))
    plt.figure(figsize=(8, 8))
    #     plt.ylim([-ylim,ylim])
    #     plt.xlim([-xlim,xlim])
    plt.scatter(sims1, sims2)
    plt.plot([0] * 100, np.linspace(-ylim, ylim, 100).tolist(), 'k--', linewidth=1)
    plt.plot(np.linspace(-xlim, xlim, 100).tolist(), [0] * 100, 'k--', linewidth=1)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    #     plt.title('Visualization of PCA')
    # for i, word in enumerate(words[:100]):
    for i, word in enumerate(words):
        plt.annotate(word, xy=(sims1[i], sims2[i]))
        
#this visualization only makes sense for two projections
def vis_variances(projections,words):
    proj1,proj2=projections.values()
    name1,name2=projections.keys()
    fig=plt.figure(figsize=(15,30))
    for enum,word in enumerate(words):
        enum=enum/2
        plt.plot([proj1[word],proj2[word]],[100-enum,100-enum],"^-")
        plt.annotate(word, xy=(min(proj2[word],proj1[word])+.5*abs(proj1[word]-proj2[word]),100-enum+.05))
        plt.annotate(name1,xy=(proj1[word],100-enum+.05))
        plt.annotate(name2,xy=(proj2[word],100-enum+.05))
    plt.xlabel('Bias Dimension' )
    title='%s_%s_filtered_normalized.png' %(name1,name2)
    plt.title(title)
    plt.show()