import matplotlib.pyplot as plt
import numpy as np
import math

def get_freq_deltas(pitches):

    #get equal temperament values
    eq_temp = np.array([ 16.352*2**(i/12) for i in range(88)])

    refs_idx = [np.argmin(np.abs(eq_temp-p)) for p in pitches]
    deltas = []
    for i,j in zip(pitches,refs_idx):
        deltas.append(1200*np.log2(i/eq_temp[j]))

    return deltas



if __name__ == "__main__":
    #read in frequencies
    g_freq = np.genfromtxt('ghana_flute_sample_f0.csv',delimiter=',')[1:,1]
    w_freq = np.genfromtxt('western_flute_sample_f0.csv',delimiter=',')[1:,1]
    
    g_deltas = get_freq_deltas(g_freq)
    w_deltas = get_freq_deltas(w_freq)

    plt.hist(g_deltas, bins=50, label='Ghana', fc=(0,0,1,0.5))
    plt.hist(w_deltas, bins=50, label='western', fc=(1,0,0,0.5))
    plt.xlabel('cent delta')
    plt.legend()
    plt.savefig('delta_histogram.png')
    plt.close()

    plt.subplot(2,1,1)
    t = np.arange(0,len(g_freq)/100,0.01)
    plt.plot(t,g_freq, label='Ghana')
    plt.plot(t,w_freq, label='western')
    plt.xlabel('time (s)')
    plt.ylabel('Hertz')
    plt.grid()
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t,g_deltas, label='Ghana')
    plt.plot(t,w_deltas, label='western')
    plt.xlabel('time (s)')
    plt.ylabel('Cents')
    plt.legend()
    plt.grid()
    plt.savefig('timeseries.png')
