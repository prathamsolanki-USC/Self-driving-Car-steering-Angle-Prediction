import matplotlib.pyplot as plt
import numpy as np
def disp(plt):
    global DISPLAY_PLOTS
    if DISPLAY_PLOTS == True:
        plt.show()
def plot_histogram(data,num_bins):
    hist, bins = np.histogram(data['steering'], num_bins)  # dividing entire range in to 25 intervals
    center = (bins[:-1] + bins[1:]) * 0.5
    # print(bins)
    plt.bar(center, hist, width=0.05)
    # plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
    disp(plt)
    return hist,bins,center


