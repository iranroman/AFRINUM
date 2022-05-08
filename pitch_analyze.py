from scipy.stats import linregress
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import scipy.stats as stats
import librosa
import librosa.display

def get_freq_deltas(pitches):

    #get equal temperament values
    eq_temp = np.array([ 16.352*2**(i/12) for i in range(88)])

    refs_idx = [np.argmin(np.abs(eq_temp-p)) for p in pitches]
    deltas = []
    for i,j in zip(pitches,refs_idx):
        deltas.append(1200*np.log2(i/eq_temp[j]))

    return deltas



if __name__ == "__main__":

    all_f0_files = os.listdir('Selected_Daddy_Lumba_Songs/vocals/crepe_output/')
    all_f0_files.sort()
    means = []
    stds = []
    years = []
    hists = []
    for f0s in all_f0_files:
        #x, fs = librosa.load('Selected_Daddy_Lumba_Songs/'+f0s.split('.')[0]+'.mp3', sr=8000)
        #X = librosa.feature.melspectrogram(x, sr=fs, n_fft=2014,win_length=256,hop_length=int(fs/100), )
        #S_db = librosa.amplitude_to_db(np.abs(X), ref=np.max)

        #plt.figure(figsize=(30,10))
        #librosa.display.specshow(S_db[:,:1000], sr=fs, hop_length=int(fs/100), x_axis='time', y_axis='mel')
        #plt.savefig('Selected_Daddy_Lumba_Songs/vocals/plots/year_{}_spec.png'.format(f0s[:4]))
        #plt.close()

        #read in frequencies
        g_freq = np.genfromtxt('Selected_Daddy_Lumba_Songs/vocals/crepe_output/'+f0s,delimiter=',')[:,1]
        conf = np.genfromtxt('Selected_Daddy_Lumba_Songs/vocals/crepe_output/'+f0s,delimiter=',')[:,2]
        
        #print(X.shape)
        #print(g_freq.shape)
        #input()

        #g_freq = [f for f,c in zip(g_freq,conf) if f<300 and f>80 and c > 0.9]
        g_freq = [(f,i) for i, (f,c) in enumerate(zip(g_freq,conf)) if f<600 and f>150 and c > 0.9]
        g_freq, time = list(map(list, zip(*g_freq)))

        g_deltas = get_freq_deltas(g_freq)
        g_cents = np.array([(1200)*np.log2(i/16.352) for i in g_freq])
        hists.append(np.histogram(g_cents,bins=range(3800,6300), density=True)[0])

        plt.hist(g_deltas, bins=50, fc=(0,0,1,0.5), density=True)
        plt.xlabel('cent delta')
        plt.savefig('Selected_Daddy_Lumba_Songs/vocals/plots/year_{}.png'.format(f0s[:4]))
        plt.close()

        means.append(np.mean(g_deltas))
        stds.append(np.std(g_deltas))
        years.append(int(f0s[:4]))

    plt.figure(figsize=(10,10))
    edges = np.arange(3800,6400,100)
    hists = np.array(hists/np.max(hists,axis=1,keepdims=True))
    plt.imshow(hists.T, aspect='auto', origin='lower', extent=[0, len(years), edges[0], edges[-1]])
    plt.yticks(edges,['D','','E','F','','G','','A','','B','C','','D','','E','F','','G','','A','','B','C','','D',''])
    plt.xticks(range(len(years)),[str(y) for y in years],rotation=45)
    plt.grid()
    plt.savefig('Selected_Daddy_Lumba_Songs/vocals/plots/time_hists.png')
    plt.close()

    years = np.array(years)
    res_means = linregress(years, means)
    res_stds = linregress(years, stds)
    plt.scatter(years,means)
    plt.plot(years, res_means.intercept + res_means.slope*years, 'r')
    plt.xlabel('years')
    plt.ylabel('mean difference from equal temperament')
    plt.grid()
    plt.title('R = {:.4f}, p = {:.4f}'.format(res_means.rvalue, res_means.pvalue))
    plt.savefig('Selected_Daddy_Lumba_Songs/vocals/plots/means_years.png')
    plt.close()
    plt.scatter(years,stds)
    plt.plot(years, res_stds.intercept + res_stds.slope*years, 'r')
    plt.xlabel('years')
    plt.ylabel('standard deviation of the difference from equal temperament')
    plt.grid()
    plt.title('R = {:.4f}, p = {:.4f}'.format(res_stds.rvalue, res_stds.pvalue))
    plt.savefig('Selected_Daddy_Lumba_Songs/vocals/plots/stds_years.png')
    plt.close()
