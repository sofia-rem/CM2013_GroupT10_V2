import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks"))

import AAAA_t10track_ecg
import adapter

track = AAAA_t10track_ecg.ECGCinC2017Track()
#track = AAAA_t10track_ecg.ECGCinC2017Track(cfg={"rpeak_method":"adaptive"})
SEED = 0
SUBSET = 2000

USE_REAL = True
if USE_REAL:
    cache_dir = PROJECT_ROOT / "data_cache" / "ecg_B0"
    print("\nDownloading ECG recordings...")
    track.download(str(cache_dir), subset = SUBSET, seed = SEED)
    records = track.load(str(cache_dir))
    track.assert_no_holdout(records)
else:
    records = track.smoke()
    print(f"Generated {len(records)} synthetic ECG recordings")


#print(records)
# records[0]
# records[0].group
# records[0].fs
# records[0].labels
# records[0].epochs

#print(records[0].group) #A00002
#print(records[0].fs) #300.0
#print(records[0].labels) #N
#print(records[0].epochs) #{'ecg' : array , shape=(1,9000)}

#print(records[0].epochs["ecg"]) # 1x1 list w. a list
#print(records[0].epochs["ecg"][0]) # 1xlength list w. ecg values

# ---------
# FEATURES
# -------------
def pNN50(r_times, fs):
    # THESE ARE R-PEAKS - NOT RRI - CORRECTED NOW - SHOULD WORK

    perc = 0
    for i in range (len(r_times)-2):
        #
        RRI1 = abs(r_times[i]-r_times[i+1])
        RRI2 = abs(r_times[i+1]-r_times[i+2])
        if 1/fs * abs(RRI1 - RRI2) > 0.05:
            perc = perc + 1
    perc = 100 * perc / (len(r_times)-2)
    return perc



# --------------
# PREPROCESS - POWERLINE
# ----------------

#uv = records[55].epochs["ecg"][0]
#print(uv)

violS = {"N":[],"A":[],"O":[],"~":[]}
violN = []
violA = []
violO = []
violNo = []

"""
for i in range(len(records)):
    #
    #print(i)
    #records[i].epochs["ecg"][0] = adapter._remove_powerline(records[i].epochs["ecg"][0],records[i].fs,how="notch")

    # --------------
    # FEATURE EXTRACTION 
    # ----------
    RRIs = AAAA_t10track_ecg._rpeaks_improved(records[i].epochs["ecg"][0],records[i].fs)
    #print(RRIs)
    print(i)

    if RRIs[0] == -76547654:
        print(i)
    

    
    pNN50_s = pNN50(RRIs)

    match records[i].labels[0]:
        case "N":
            violN.append(pNN50_s)
        case "A":
            violA.append(pNN50_s)
        case "O":
            violA.append(pNN50_s)
        case "~":
            violA.append(pNN50_s)

"""
# -----------
# VIOLIN PLOT OF pNN50
# ---------
#print(violN)
#print(records[55].epochs["ecg"][0])


# -------------
# DO POWER SPECTRUM HERE
# -------------
import scipy
import matplotlib


# PLOT OF ECG W VERTICAL LINES FOR DETECTED HEART BEATS
"""
i = 1944
#records[i].epochs["ecg"][0] = adapter._remove_powerline(records[i].epochs["ecg"][0],records[i].fs,how="notch")
#records[i].epochs["ecg"][0] = adapter._remove_baseline(records[i].epochs["ecg"][0],records[i].fs,how="highpass",order=6)
#RRIs = AAAA_t10track_ecg._rpeaks_im_V2(records[i].epochs["ecg"][0],records[i].fs)
RRIs = AAAA_t10track_ecg._try_new_rpeaks(records[i].epochs["ecg"][0],records[i].fs)
RRIs_sec = 1/300 * RRIs
rec_derivative = np.gradient(records[i].epochs["ecg"][0])
rec_der_exp = 20*rec_derivative
fig, (axA,axB,axC) = matplotlib.pyplot.subplots(3)
axA.plot(records[i].epochs["ecg"][0])
axC.plot(rec_der_exp)
axA.vlines(RRIs,0,1.,colors="red")
axA.set_title("track 1944")

i = 1655
#records[i].epochs["ecg"][0] = adapter._remove_powerline(records[i].epochs["ecg"][0],records[i].fs,how="notch")
#records[i].epochs["ecg"][0] = adapter._remove_baseline(records[i].epochs["ecg"][0],records[i].fs,how="highpass",order=6)
#RRIs = AAAA_t10track_ecg._rpeaks_im_V2(records[i].epochs["ecg"][0],records[i].fs)
RRIs = AAAA_t10track_ecg._try_new_rpeaks(records[i].epochs["ecg"][0],records[i].fs)
RRIs_sec = 1/300 * RRIs
rec_derivative = np.gradient(records[i].epochs["ecg"][0])
rec_der_exp = 20*rec_derivative
axB.plot(records[i].epochs["ecg"][0])
axC.plot(rec_der_exp)
axB.vlines(RRIs,0,1.,colors="red")
axB.set_title("track 1655")

i = 1166
#records[i].epochs["ecg"][0] = adapter._remove_powerline(records[i].epochs["ecg"][0],records[i].fs,how="notch")
#records[i].epochs["ecg"][0] = adapter._remove_baseline(records[i].epochs["ecg"][0],records[i].fs,how="highpass",order=6)
#RRIs = AAAA_t10track_ecg._rpeaks_im_V2(records[i].epochs["ecg"][0],records[i].fs)
#RRIs = AAAA_t10track_ecg._rpeaks_im_V2(records[i].epochs["ecg"][0],records[i].fs)
RRIs = AAAA_t10track_ecg._try_new_rpeaks(records[i].epochs["ecg"][0],records[i].fs)
RRIs_sec = 1/300 * RRIs
rec_derivative = np.gradient(records[i].epochs["ecg"][0])
rec_der_exp = 20*rec_derivative
axC.plot(records[i].epochs["ecg"][0])
#axC.plot(rec_derivative)
axC.plot(rec_der_exp)
axC.vlines(RRIs,0,1.,colors="red")
axC.set_title("track 1166")
"""


fig , ax = matplotlib.pyplot.subplots(3)
#is_ii = [1944, 1655, 1166]
is_ii = [1,2,3]
for ii in range(len(is_ii)):
    print(records[ii].labels)
    i = is_ii[ii]
    #RRIs = AAAA_t10track_ecg._try_new_rpeaks(records[i].epochs["ecg"][0],records[i].fs)
    #RRIs = AAAA_t10track_ecg._rpeaks_improved(records[i].epochs["ecg"][0],records[i].fs)
    RRIs = AAAA_t10track_ecg._divisible_len(records[i].epochs["ecg"][0],records[i].fs)
    RRIs_sec = 1/300 * RRIs
    rec_derivative = 20 * np.gradient(records[i].epochs["ecg"][0])
    ax[ii].vlines(RRIs, 0,2,color="red")
    ax[ii].plot(records[i].epochs["ecg"][0],color="blue")
    #ax[ii].plot(rec_derivative,color="orange")
    RRIs_yval = np.ones(len(RRIs))
    #ax[ii].scatter(RRIs,RRIs_yval)



    



matplotlib.pyplot.show()


j = np.array([1,2,3,4,6,7,4,2,99])
j2 = []

for i in range(4,8):
    j2.append(j[i])

print(j2)

j3 = j.tolist()
j4 = j3[4:8]
print(j4) #works
j5 = np.array(j4)

"""
PSD_Na = [0]*129
PSD_Aa = [0]*129
PSD_Oa = [0]*129
PSD_Noi = [0]*129
#print(PSD_Na)

fig, axS = matplotlib.pyplot.subplots(4,2)


for i in range(len(records)):
    # 
    #the_rec = records[i].epochs["ecg"][0]
    #the_lab = records[i].labels[0]
    #print(the_rec , the_lab)
    freq, PSD = scipy.signal.welch(records[i].epochs["ecg"][0], records[0].fs)
    #print(freq[0] , freq[-1] , len(freq) , len(PSD))

    if records[i].labels[0] == "N":

        axS[0,0].semilogy(freq,PSD)
        PSD_Na = [x+y for x,y in zip(PSD_Na,PSD)]

    if records[i].labels[0] == "A":
    
        axS[0,1].semilogy(freq,PSD)
        PSD_Aa = [x+y for x,y in zip(PSD_Aa,PSD)]

    if records[i].labels[0] == "O":
    
        axS[1,0].semilogy(freq,PSD)
        PSD_Oa = [x+y for x,y in zip(PSD_Oa,PSD)]

    if records[i].labels[0] == "~":
    
        axS[1,1].semilogy(freq,PSD)
        PSD_Noi = [x+y for x,y in zip(PSD_Noi,PSD)]



axS[0,0].set_title("Normal rythm")
axS[0,1].set_title("AF")
axS[1,0].set_title("Other rythm")
axS[1,1].set_title("Noise")

axS[2,0].semilogy(freq,PSD_Na)
axS[2,0].set_title("sum of normals")
axS[2,1].semilogy(freq,PSD_Aa)
axS[2,1].set_title("sum of AF")
axS[3,0].semilogy(freq,PSD_Oa)
axS[3,0].set_title("sum of others")
axS[3,1].semilogy(freq,PSD_Noi)
axS[3,1].set_title("sum of noise")


matplotlib.pyplot.show()
"""


print("done")