from scipy.stats import linregress
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import scipy.stats as stats
import librosa
import librosa.display
import pickle
#from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture as GMM
from sklearn import metrics
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

#with (open("comps.pkl", "rb")) as openfile:
#    Cs = pickle.load(openfile)


if __name__ == "__main__":

    # load the CREPE outputs
    all_f0_files = os.listdir('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/crepe_output/')
    all_f0_files.sort()

    # compute information about years
    years_float = [float(f[:4]) for f in all_f0_files]
    years_diff = 1+ -1*np.sign(np.insert(np.diff(years_float),0,1))
    cum_sum = 0
    for i,v in enumerate(years_diff):
        if v != 0:
            cum_sum += 1
        else:
            cum_sum = 0
        years_diff[i] = cum_sum
    years_diff = [v/10 for v in years_diff]
    years_float = np.array(years_float) + np.array(years_diff)


    means = []
    stds = []
    years = []
    hists = []
    ncomps = []
    comps = []
    scores = []
    covariances = []
    new_year = 0
    for i, f0s in enumerate(all_f0_files):
        print(f0s)
        year = f0s[:4]+'  '+str(i+1)
        g_freq = np.genfromtxt('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/crepe_output/'+f0s,delimiter=',')[:,1]
        conf = np.genfromtxt('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/crepe_output/'+f0s,delimiter=',')[:,2]
        

        # filter out pitches outside the male range of singing and with low CREPE confidence
        orig_len = len(g_freq)
        g_freq = [(f,i) for i, (f,c) in enumerate(zip(g_freq,conf)) if f<600 and f>82.4 and c > 0.8]
        g_freq, time = list(map(list, zip(*g_freq)))

        # convert to cents
        g_cents = np.array([(1200)*np.log2(i/16.352) for i in g_freq])
        # keep track of all histograms
        hists.append(np.histogram(g_cents,bins=range(2800,6300), density=True)[0])
        most_common_note = np.argmax(hists[-1])+2800

        # limiting to one octave around the most common note
        g_cents = np.array([c for c in g_cents if c>(most_common_note-600) and c<(most_common_note+600])
        g_cents = g_cents[...,np.newaxis]

        # find the optimal number of F0 components and their associated mean and covariance
        embeddings = g_cents
        #n_clusters= [len(Cs[i])]
        n_clusters= range(4,15)
        iterations=10
        best_score = 0.0
        components = []
        covariance = []
        for n in n_clusters:
            for _ in range(iterations):
                gmm=GMM(n,covariance_type='tied').fit(embeddings) 
                labels=gmm.predict(embeddings)
                sil=metrics.silhouette_score(embeddings, labels, metric='euclidean')
                if sil > best_score:
                    best_score = sil
                    components = np.sort(np.squeeze(gmm.means_))
                    covariance = np.sqrt(np.squeeze(gmm.covariances_))

        comps.append(components)
        ncomps.append(len(components))
        scores.append(best_score)
        covariances.append(covariance)

        if len(years)>0 and year.split('  ')[0]==new_year:
            years.append(year.split('  ')[1])
        else:
            years.append(year)
        new_year = f0s[:4]
        print(f0s)
        print(comps[-1])
        print(scores[-1])
        print(covariances[-1])
        print(years[-1])
        print('')

    np.save('comps.npy',np.array(comps))
    np.save('scores.npy',np.array(scores))
    np.save('covariances.npy',np.array(covariances))
    np.save('years.npy',np.array(years))

    # code to load the pre-computed components
    comps = np.load('comps.npy', allow_pickle=True)
    scores = np.load('scores.npy', allow_pickle=True)
    covariances = np.load('covariances.npy', allow_pickle=True)
    years = np.load('years.npy', allow_pickle=True)
    curr_year = 0
    years_float = []
    for y in years:
        year = float(y[:4])
        if year > curr_year:
            curr_year = year
        years_float.append(curr_year)
    years_float = np.array(years_float)


    res_means = linregress(years_float, covariances)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(years_float,covariances,color='k')
    plt.plot(years_float, res_means.intercept + res_means.slope*years_float, 'r')
    plt.title('r = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.grid(linestyle='dotted')
    plt.ylabel('std dev of Gaussian comps')
    plt.xlabel('year')
    ncomps = np.array([len(c) for c in comps])
    res_means = linregress(years_float, ncomps)
    plt.subplot(1,2,2)
    plt.scatter(years_float,ncomps,color='k')
    plt.plot(years_float, res_means.intercept + res_means.slope*years_float, 'r')
    plt.title('r = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.grid(linestyle='dotted')
    plt.ylabel('number of Gaussian comps')
    plt.xlabel('year')
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/cov_reg.png')
    plt.close()
    print('covariance regression plotted!')

    import itertools as it
    hists_max = [np.argmax(h)+2800 for h in hists]
    distances = []
    for j, c in enumerate(comps):
        c = [i for i in c if i>hists_max[j]-600 and i<hists_max[j]+600]
        all_note_dists = [y - x for x, y in it.combinations(c, 2)]
        all_mod = [d%100 for d in all_note_dists]
        all_mod_half = [d if d<50 else (100-d) for d in all_mod]
        distances.append(all_mod_half)


    ncomps = np.array([np.std(dists) for dists in distances])
    res_means = linregress(years_float, ncomps)
    plt.scatter(years_float,ncomps,color='k')
    plt.plot(years_float, res_means.intercept + res_means.slope*years_float, 'r')
    plt.title('r = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.grid(linestyle='dotted')
    plt.ylabel('ε$_s$ (in cents)')
    plt.xlabel('year')
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/dists_reg.png')
    plt.close()
    print('regression over comp distances plotted!')
    


    plt.figure(figsize=(22,10))
    edges = np.arange(2800,6400,100)
    hists = np.array(hists/np.max(hists,axis=1,keepdims=True))
    plt.imshow(hists.T, aspect='auto', origin='lower', extent=[0, len(years), edges[0], edges[-1]])
    for i in range(len(years)):
        plt.scatter([i+0.5]*len(comps[i]),comps[i], color='white', edgecolor='k',s=18)
    plt.yticks(edges,['E2 (2800)','F (2900)','(3000)','G (3100)','(3200)','A (3300)','(3400)','B (3500)','C3 (3600)','(3700)','D (3800)','(3900)','E (4000)','F (4100)','(4200)','G (4300)','(4400)','A (4500)','(4600)','B (4700)','C3 (4800)','(4900)','D (5000)','(5100)','E (5200)','F (5300)','(5400)','G (5500)','(5600)','A (5700)','(5800)','B (5900)','C4 (6000)','(6100)','D (6200)','(6300)'])
    plt.xticks(range(len(years)),[str(y) for y in years],rotation=45)
    plt.grid(linestyle='dotted')
    plt.ylim([4000,6000])
    plt.ylabel('F0 value in cents (reference is 16.35Hz C0)')
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/time_hists.png')
    plt.close()
