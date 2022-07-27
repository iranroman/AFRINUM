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

def get_freq_deltas(pitches):

    #get equal temperament values
    eq_temp = np.array([ 16.352*2**(i/12) for i in range(88)])

    refs_idx = [np.argmin(np.abs(eq_temp-p)) for p in pitches]
    deltas = []
    for i,j in zip(pitches,refs_idx):
        deltas.append(1200*np.log2(i/eq_temp[j]))

    return deltas


if __name__ == "__main__":

    all_f0_files = os.listdir('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/crepe_output/')
    all_f0_files.sort()
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
    for i, f0s in enumerate(all_f0_files):
        year = f0s[:4]+'  '+str(i+1)
        #x, fs = librosa.load('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/'+f0s.split('.')[0]+'.mp3', sr=8000)
        #X = librosa.feature.melspectrogram(x, sr=fs, n_fft=2014,win_length=256,hop_length=int(fs/100), )
        #S_db = librosa.amplitude_to_db(np.abs(X), ref=np.max)

        #plt.figure(figsize=(30,10))
        #librosa.display.specshow(S_db[:,:1000], sr=fs, hop_length=int(fs/100), x_axis='time', y_axis='mel')
        #plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/vocals/plots/year_{}_spec.png'.format(f0s[:4]))
        #plt.close()

        #read in frequencies
        g_freq = np.genfromtxt('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/crepe_output/'+f0s,delimiter=',')[:,1]
        conf = np.genfromtxt('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/crepe_output/'+f0s,delimiter=',')[:,2]
        
        #print(X.shape)
        #print(g_freq.shape)
        #input()

        #g_freq = [f for f,c in zip(g_freq,conf) if f<300 and f>80 and c > 0.9]
        g_freq = [(f,i) for i, (f,c) in enumerate(zip(g_freq,conf)) if f<600 and f>80 and c > 0.8] 
        g_freq, time = list(map(list, zip(*g_freq)))


        #g_deltas = get_freq_deltas(g_freq)
        g_cents = np.array([(1200)*np.log2(i/16.352) for i in g_freq])
        hists.append(np.histogram(g_cents,bins=range(2800,6300), density=True)[0])

        g_cents = g_cents[...,np.newaxis]


        embeddings = g_cents
        n_clusters=np.arange(4, 30) # the highest number of components found in the corpus
        iterations=20
        best_score = 0.0
        components = []
        covariance = []
        for n in n_clusters:
            tmp_sil=[]
            tmp_cov=[]
            tmp_comps=[]
            for _ in range(iterations):
                gmm=GMM(n,covariance_type='tied').fit(embeddings) 
                labels=gmm.predict(embeddings)
                sil=metrics.silhouette_score(embeddings, labels, metric='euclidean')
                tmp_sil.append(sil)
                tmp_cov.append(np.squeeze(gmm.covariances_))
                tmp_comps.append(np.sort(np.squeeze(gmm.means_))[np.newaxis])
            val=np.mean(tmp_sil)
            cov=np.sqrt(np.median(tmp_cov))
            if val > best_score:
                #comp=np.median(np.concatenate(tmp_comps,axis=0),axis=0)
                comp=np.median(np.concatenate([tmp_comps[-1]],axis=0),axis=0)
                best_score = val
                components = np.delete(comp, np.argwhere(np.ediff1d(comp) <= 30) + 1) # remove reduntant components
                covariance = cov

        print(f0s)
        print(components)
        print('ncomps:',len(components))
        print('cov:',covariance)
        pts = components[:,np.newaxis]
        dis = list(set(list(np.abs(pts[np.newaxis, :, :] - pts[:, np.newaxis, :]).min(axis=2).flatten())))
        #print(['{:.2f}'.format(i) for i in dis])
        dis = [i%100 if (i%100)<50 else (i%100)-100 for i in dis if i != 0]
        dis.sort()
        print(['{:.2f}'.format(i) for i in dis])
        print(np.mean(np.abs(dis)))
        print(np.mean(dis))
        print(np.std(np.abs(dis)))
        print(np.std(dis))
        print(' ')

        comps.append(components)
        ncomps.append(len(components))
        scores.append(best_score)
        covariances.append(covariance)

open_file = open('comps.pkl', "wb")
pickle.dump(comps, open_file)
open_file.close()
open_file = open('scores.pkl', "wb")
pickle.dump(scores, open_file)
open_file.close()
open_file = open('covariances.pkl', "wb")
pickle.dump(covariances, open_file)
open_file.close()

'''
        comps.append(components)
        ncomps.append(len(components))
        scores.append(best_score)
        covariances.append(covariance)
        print(comps[-1])
        print(scores[-1])
        print(covariances[-1])
        print('')

        plt.hist(g_deltas, bins=50, fc=(0,0,1,0.5), density=True)
        plt.xlabel('cent delta')
        plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/year_{}.png'.format(year))
        plt.close()

        means.append(np.mean(g_deltas))
        stds.append(np.std(g_deltas))
        years.append(year)

    res_means = linregress(years_float, covariances)
    plt.scatter(years_float,covariances,color='k')
    plt.plot(years_float, res_means.intercept + res_means.slope*years_float, 'r')
    plt.title('R = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.grid(linestyle='dotted')
    plt.ylabel('std dev of Gaussian comps')
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/cov_reg.png')
    plt.close()
    print(scores)


    res_means = linregress(years_float, ncomps)
    plt.scatter(years_float, ncomps, color='k')
    plt.plot(years_float, res_means.intercept + res_means.slope*years_float, 'r')
    plt.title('R = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.grid(linestyle='dotted')
    plt.ylabel('No. of Gaussian comps')
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/num_comps.png')
    plt.close()

    plt.figure(figsize=(22,10))
    edges = np.arange(2800,6400,100)
    hists = np.array(hists/np.max(hists,axis=1,keepdims=True))
    plt.imshow(hists.T, aspect='auto', origin='lower', extent=[0, len(years), edges[0], edges[-1]])
    for i in range(len(years)):
        plt.scatter([i+0.5]*len(comps[i]),comps[i], color='white', edgecolor='k')
    plt.yticks(edges,['E2 (2800)','F (2900)','(3000)','G (3100)','(3200)','A (3300)','(3400)','B (3500)','C3 (3600)','(3700)','D (3800)','(3900)','E (4000)','F (4100)','(4200)','G (4300)','(4400)','A (4500)','(4600)','B (4700)','C3 (4800)','(4900)','D (5000)','(5100)','E (5200)','F (5300)','(5400)','G (5500)','(5600)','A (5700)','(5800)','B (5900)','C4 (6000)','(6100)','D (6200)','(6300)'])
    plt.xticks(range(len(years)),[str(y) for y in years],rotation=45)
    plt.grid(linestyle='dotted')
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/time_hists.png')
    plt.close()

    #years = np.array(years)
    res_means = linregress(years_float, np.abs(means))
    res_stds = linregress(years_float, stds)
    plt.scatter(years_float,np.abs(means), color='k')
    plt.plot(years_float, res_means.intercept + res_means.slope*years_float, 'r')
    plt.xlabel('years')
    plt.ylabel('mean difference from equal temperament')
    plt.grid(linestyle='dotted')
    plt.title('R = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/means_years.png')
    plt.close()

    plt.scatter(years_float,stds, color='k')
    plt.plot(years_float, res_stds.intercept + res_stds.slope*years_float, 'r')
    plt.xlabel('years')
    plt.ylabel('std dev of the difference from equal temperament')
    plt.grid(linestyle='dotted')
    plt.title('R = {:.4f}, p = {:.4f}'.format(res_stds.rvalue, res_stds.pvalue))
    plt.savefig('ALL_DADDY_LUMBA_TRACKS_IN_ORDER/plots/stds_years.png')
    plt.close()
'''
