#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne as mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

eeg=mne.io.read_raw_brainvision("C:/Users/Anjali Singh/Downloads/eegfmri/alex/alex_rest.vhdr",preload=True)
eeg_r1=mne.io.read_raw_brainvision("C:/Users/Anjali Singh/Downloads/eegfmri/alex/alex_retino_gamma_01.vhdr",preload=True)
eeg_r2=mne.io.read_raw_brainvision("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_02.vhdr",preload=True)


# In[2]:


data_a=eeg.get_data() #raw data
events_a=mne.events_from_annotations(eeg) #evts changed to events

#Retino-1
data_r1=eeg_r1.get_data() 
events_r1=mne.events_from_annotations(eeg_r1) 

#Retino-2
data_r2=eeg_r2.get_data() #raw data
events_r2=mne.events_from_annotations(eeg_r2)


# In[3]:


vol_trigs_a=events_a[1]['Response/R128']
vol_inds_a=np.argwhere(events_a[0][:,2]==vol_trigs_a)
vol_times_a=events_a[0][vol_inds_a,0].astype(int) #these are times corresponding to those indices

#Retino-1
vol_trigs_r1=events_r1[1]['Response/R128']
vol_inds_r1=np.argwhere(events_r1[0][:,2]==vol_trigs_r1)
vol_times_r1=events_r1[0][vol_inds_r1,0].astype(int) 

#Retino-2
vol_trigs_r2=events_r2[1]['Response/R128']
vol_inds_r2=np.argwhere(events_r2[0][:,2]==vol_trigs_r2)
vol_times_r2=events_r2[0][vol_inds_r2,0].astype(int) #these are times corresponding to those indices


# In[4]:


#Step 2-Part B
#removing gradient artifact which is caused by scanner while ballistographic is caused by heartbeat
grad_len=3465
grad_epochs_a=np.reshape(data_a[:,vol_times_a[0,0]:vol_times_a[-2,0]+grad_len],[64,vol_times_a.shape[0]-1,grad_len])
grad_epochs_a.shape

grad_len=3465
grad_epochs_r1=np.reshape(data_r1[:,vol_times_r1[0,0]:vol_times_r1[-2,0]+grad_len],[64,vol_times_r1.shape[0]-1,grad_len])

grad_len=3465
grad_epochs_r2=np.reshape(data_r2[:,vol_times_r2[0,0]:vol_times_r2[-2,0]+grad_len],[64,vol_times_r2.shape[0]-1,grad_len])


# In[5]:


#subtracted equals
subbed_a=grad_epochs_a-np.expand_dims(np.mean(grad_epochs_a,axis=1),axis=1)
subbed_a=np.reshape(subbed_a,data_a[:,vol_times_a[0,0]:vol_times_a[-2,0]+grad_len].shape)

subbed_r1=grad_epochs_r1-np.expand_dims(np.mean(grad_epochs_r1,axis=1),axis=1)
subbed_r1=np.reshape(subbed_r1,data_r1[:,vol_times_r1[0,0]:vol_times_r1[-2,0]+grad_len].shape)

subbed_r2=grad_epochs_r2-np.expand_dims(np.mean(grad_epochs_r2,axis=1),axis=1)
subbed_r2=np.reshape(subbed_r2,data_r2[:,vol_times_r2[0,0]:vol_times_r2[-2,0]+grad_len].shape)

fig,axes=plt.subplots(3,1,figsize=(10,8))
axes[0].plot(subbed_a[3,:])
axes[1].plot(subbed_r1[3,:])
axes[2].plot(subbed_r2[3,:])
axes[0].set_title("Alex-rest")
axes[1].set_title("Alex-retino-1")
axes[2].set_title("Alex-retino-2")
plt.tight_layout()


# In[6]:


#plotting not subtracted and subtracted eeg
not_sub_a=np.abs(np.fft.fft(data_a[3,vol_times_a[0,0]:vol_times_a[-2,0]+grad_len]))
sub_a=np.abs(np.fft.fft(subbed_a[3,:]))

not_sub_r1=np.abs(np.fft.fft(data_r1[3,vol_times_r1[0,0]:vol_times_r1[-2,0]+grad_len]))
sub_r1=np.abs(np.fft.fft(subbed_r1[3,:]))

not_sub_r2=np.abs(np.fft.fft(data_r2[3,vol_times_r2[0,0]:vol_times_r2[-2,0]+grad_len]))
sub_r2=np.abs(np.fft.fft(subbed_r2[3,:]))

fig,axes=plt.subplots(3,2,figsize=(10,8))
axes[0][0].plot(not_sub_a); axes[0][1].plot(sub_a);
axes[1][0].plot(not_sub_r1); axes[1][1].plot(sub_r1);
axes[2][0].plot(not_sub_r2); axes[2][1].plot(sub_r2);

axes[0][0].set_title("Alex-rest--not subtracted"); axes[0][1].set_title("Alex-rest-subtracted")
axes[1][0].set_title("Alex-retino1-not subtracted"); axes[1][1].set_title("Alex-retino1-not subtracted")
axes[2][0].set_title("Alex-retino2-not subtracted"); axes[2][1].set_title("Alex-retino2-not subtracted")
plt.tight_layout()


# In[7]:


#Step 2- Part B 
#BCG artifact removal
from scipy.signal import resample
from scipy.signal import find_peaks

#downsampling from 5000 Hz to 250 Hz
resamp_a=resample(subbed_a,int(subbed_a.shape[1]/(5000/250)),axis=1)
resamp_r1=resample(subbed_r1,int(subbed_r1.shape[1]/(5000/250)),axis=1)
resamp_r2=resample(subbed_r2,int(subbed_r2.shape[1]/(5000/250)),axis=1)

#Finding peaks
peaks_a=find_peaks(resamp_a[31,:],distance=180)
peaks_r1=find_peaks(resamp_r1[31,:],distance=180)
peaks_r2=find_peaks(resamp_r2[31,:],distance=180)


# In[8]:


#epoch the data using peaks
eps_a=np.zeros([64,peaks_a[0].shape[0],210])
for i in np.arange(1,peaks_a[0].shape[0]-1):
    eps_a[:,i,:]=resamp_a[:,peaks_a[0][i]-105:peaks_a[0][i]+105]
    
eps_r1=np.zeros([64,peaks_r1[0].shape[0],210])
for i in np.arange(1,peaks_r1[0].shape[0]-1):
    eps_r1[:,i,:]=resamp_r1[:,peaks_r1[0][i]-105:peaks_r1[0][i]+105]
    
eps_r2=np.zeros([64,peaks_r2[0].shape[0],210])
for i in np.arange(1,peaks_r2[0].shape[0]-1):
    eps_r2[:,i,:]=resamp_r2[:,peaks_r2[0][i]-105:peaks_r2[0][i]+105]


# In[9]:


#epoch the data using peaks
mean_bcg_a=np.mean(eps_a,axis=1)
bcg_sub_a=np.copy(resamp_a)
for i in np.arange(1,peaks_a[0].shape[0]-1):
    bcg_sub_a[:,peaks_a[0][i]-105:peaks_a[0][i]+105]=resamp_a[:,peaks_a[0][i]-105:peaks_a[0][i]+105]-mean_bcg_a
    
mean_bcg_r1=np.mean(eps_r1,axis=1)
bcg_sub_r1=np.copy(resamp_r1)
for i in np.arange(1,peaks_r1[0].shape[0]-1):
    bcg_sub_r1[:,peaks_r1[0][i]-105:peaks_r1[0][i]+105]=resamp_r1[:,peaks_r1[0][i]-105:peaks_r1[0][i]+105]-mean_bcg_r1
    
mean_bcg_r2=np.mean(eps_r2,axis=1)
bcg_sub_r2=np.copy(resamp_r2)
for i in np.arange(1,peaks_r2[0].shape[0]-1):
    bcg_sub_r2[:,peaks_r2[0][i]-105:peaks_r2[0][i]+105]=resamp_r2[:,peaks_r2[0][i]-105:peaks_r2[0][i]+105]-mean_bcg_r2
    

fig,axes=plt.subplots(3,2,figsize=(10,8))
axes[0][0].plot(resamp_a[3,:]); axes[0][1].plot(bcg_sub_a[3,:]);
axes[1][0].plot(resamp_r1[3,:]); axes[1][1].plot(bcg_sub_r1[3,:]);
axes[2][0].plot(resamp_r2[3,:]); axes[2][1].plot(bcg_sub_r2[3,:]);

axes[0][0].set_title("Alex-rest"); axes[0][1].set_title("Alex-rest-after BCG removal")
axes[1][0].set_title("Alex-retino1"); axes[1][1].set_title("Alex-retino1-after BCG removal")
axes[2][0].set_title("Alex-retino2"); axes[2][1].set_title("Alex-retino2-after BCG removal")
plt.tight_layout()


# In[10]:


#Step2-Part C  ALEX
#ICA to isolate good and bad components
#preprocessing

#Rest
eeg.resample(250)
raw_data_a=eeg.get_data()
evts_a=mne.events_from_annotations(eeg)
vol_trigs_a=evts_a[1]['Response/R128']
vol_inds_a=np.argwhere(evts_a[0][:,2]==vol_trigs_a)
vol_times_a=evts_a[0][vol_inds_a,0].astype(int)
start_a=vol_times_a[0][0]
raw_data_a[:,start_a:start_a+bcg_sub_a.shape[1]]=bcg_sub_a
raw_data_a[:,0:start_a]=0
raw_data_a[:,start_a+bcg_sub_a.shape[1]:]=0
#overwriting eeg data with denoised data
eeg._data=raw_data_a
montage_a=mne.channels.make_standard_montage('standard_1020')
eeg.set_montage(montage_a,on_missing='ignore')

#Retino-1
eeg_r1.resample(250)
raw_data_r1=eeg_r1.get_data()
evts_r1=mne.events_from_annotations(eeg_r1)
vol_trigs_r1=evts_r1[1]['Response/R128']
vol_inds_r1=np.argwhere(evts_r1[0][:,2]==vol_trigs_r1)
vol_times_r1=evts_r1[0][vol_inds_r1,0].astype(int)
start_r1=vol_times_r1[0][0]
raw_data_r1[:,start_r1:start_r1+bcg_sub_r1.shape[1]]=bcg_sub_r1
raw_data_r1[:,0:start_r1]=0
raw_data_r1[:,start_r1+bcg_sub_r1.shape[1]:]=0
eeg_r1._data=raw_data_r1
montage_r1=mne.channels.make_standard_montage('standard_1020')
eeg_r1.set_montage(montage_a,on_missing='ignore')

#Retino-2
eeg_r2.resample(250)
raw_data_r2=eeg_r2.get_data()
evts_r2=mne.events_from_annotations(eeg_r2)
vol_trigs_r2=evts_r2[1]['Response/R128']
vol_inds_r2=np.argwhere(evts_r2[0][:,2]==vol_trigs_r2)
vol_times_r2=evts_r2[0][vol_inds_r2,0].astype(int)
start_r2=vol_times_r2[0][0]
raw_data_r2[:,start_r2:start_r2+bcg_sub_r2.shape[1]]=bcg_sub_r2
raw_data_r2[:,0:start_r2]=0
raw_data_r2[:,start_r2+bcg_sub_r2.shape[1]:]=0
eeg_r2._data=raw_data_r2
montage_r2=mne.channels.make_standard_montage('standard_1020')
eeg_r2.set_montage(montage_a,on_missing='ignore')


# In[11]:


#Performing high pass filter to remove low frequency drifts
eeg.filter(1,124)
ica_a=mne.preprocessing.ICA(n_components=60) #to define how many components we want to decompose into: n_components
good_ch_a=np.arange(0,63)
good_ch_a=np.delete(good_ch_a,20)
ica_a.fit(eeg,picks=good_ch_a)

#Retino-1
eeg_r1.filter(1,124)
ica_r1=mne.preprocessing.ICA(n_components=60) #to define how many components we want to decompose into: n_components
good_ch_r1=np.arange(0,63)
good_ch_r1=np.delete(good_ch_r1,20)
ica_r1.fit(eeg_r1,picks=good_ch_r1)

#Retino-2
eeg_r2.filter(1,124)
ica_r2=mne.preprocessing.ICA(n_components=60) #to define how many components we want to decompose into: n_components
good_ch_r2=np.arange(0,63)
good_ch_r2=np.delete(good_ch_r2,20)
ica_r2.fit(eeg_r2,picks=good_ch_r2)


# In[12]:


#Visualising the components- ALEX
ica_a.plot_components()
ica_a.plot_sources(eeg)
sources_a=ica_a.get_sources(eeg)._data


# In[13]:


#Retino-1
ica_r1.plot_components()
ica_r1.plot_sources(eeg_r1)
sources_r1=ica_r1.get_sources(eeg_r1)._data


# In[14]:


#Retino-2
ica_r2.plot_components()
ica_r2.plot_sources(eeg_r2)
sources_r2=ica_r2.get_sources(eeg_r2)._data


# In[15]:


#Plotting power spectral density of each source- ALEX

from scipy.signal import welch
psd_a=welch(sources_a,fs=250,nperseg=500)
fig, axes = plt.subplots(6, 10, figsize=(20, 10))  # Adjust the width and height as desired
for i in np.arange(0, 60):
    ax = axes[i // 10, i % 10]
    ax.plot(psd_a[0][0:120], np.log(psd_a[1][i, 0:120]), 'r')
    ax.axvline(8)
    ax.axvline(13)
    ax.set_title(i)
    
plt.subplots_adjust(hspace=2.5, wspace=0.25)  # Adjust the horizontal and vertical spacing
plt.show()


# In[16]:


#Plotting power spectral density of each source- ALEX-retino-1

from scipy.signal import welch
psd_r1=welch(sources_r1,fs=250,nperseg=500)
fig, axes = plt.subplots(6, 10, figsize=(20, 10))  # Adjust the width and height as desired
for i in np.arange(0, 60):
    ax = axes[i // 10, i % 10]
    ax.plot(psd_r1[0][0:120], np.log(psd_r1[1][i, 0:120]), 'r')
    ax.axvline(8)
    ax.axvline(13)
    ax.set_title(i)
    
plt.subplots_adjust(hspace=2.5, wspace=0.25)  # Adjust the horizontal and vertical spacing
plt.show()


# In[17]:


#Plotting power spectral density of each source- ALEX-retino-2

from scipy.signal import welch
psd_r2=welch(sources_r2,fs=250,nperseg=500)
fig, axes = plt.subplots(6, 10, figsize=(20, 10))  # Adjust the width and height as desired
for i in np.arange(0, 60):
    ax = axes[i // 10, i % 10]
    ax.plot(psd_r2[0][0:120], np.log(psd_r2[1][i, 0:120]), 'r')
    ax.axvline(8)
    ax.axvline(13)
    ax.set_title(i)
    
plt.subplots_adjust(hspace=2.5, wspace=0.25)  # Adjust the horizontal and vertical spacing
plt.show()


# In[18]:


good_comp_a=sources_a[[3,10,19],:] #components that have peak in 8-13 HZ band i.e., alpha band are good components
#alpha components
alpha_comp_a=mne.filter.filter_data(good_comp_a,250,8,13)
alpha_comp_a=np.abs(alpha_comp_a)
gamma_comp_a=mne.filter.filter_data(good_comp_a,250,40,60)
gamma_comp_a=np.abs(gamma_comp_a)
from scipy.ndimage import gaussian_filter1d
smthalphacomp_a=gaussian_filter1d(alpha_comp_a,250,axis=1)
trim_alpha_a=np.mean(smthalphacomp_a[:,start_a:],axis=0) 
#gamma components
smthgamma_comp_a=gaussian_filter1d(gamma_comp_a,250,axis=1)
trim_gamma_a=np.mean(smthgamma_comp_a[:,start_a:],axis=0)

#Retino-1
good_comp_r1=sources_r1[[7,16,19],:]
#alpha components
alpha_comp_r1=mne.filter.filter_data(good_comp_r1,250,8,13)
alpha_comp_r1=np.abs(alpha_comp_r1)
gamma_comp_r1=mne.filter.filter_data(good_comp_r1,250,40,60)
gamma_comp_r1=np.abs(gamma_comp_r1)
from scipy.ndimage import gaussian_filter1d
smthalphacomp_r1=gaussian_filter1d(alpha_comp_r1,250,axis=1)
trim_alpha_r1=np.mean(smthalphacomp_r1[:,start_r1:],axis=0) 
#gamma components
smthgamma_comp_r1=gaussian_filter1d(gamma_comp_r1,250,axis=1)
trim_gamma_r1=np.mean(smthgamma_comp_r1[:,start_r1:],axis=0)

#Retino-2
good_comp_r2=sources_r2[[10,17,23],:]
#alpha components
alpha_comp_r2=mne.filter.filter_data(good_comp_r2,250,8,13)
alpha_comp_r2=np.abs(alpha_comp_r2)
gamma_comp_r2=mne.filter.filter_data(good_comp_r2,250,40,60)
gamma_comp_r2=np.abs(gamma_comp_r2)
from scipy.ndimage import gaussian_filter1d
smthalphacomp_r2=gaussian_filter1d(alpha_comp_r2,250,axis=1)
trim_alpha_r2=np.mean(smthalphacomp_r2[:,start_r2:],axis=0) 
#gamma components
smthgamma_comp_r2=gaussian_filter1d(gamma_comp_r2,250,axis=1)
trim_gamma_r2=np.mean(smthgamma_comp_r2[:,start_r2:],axis=0)


# In[19]:


#downsampling alpha-ALEX
n_timepts_a=vol_times_a.shape[0] #450
resalpha_a=resample(trim_alpha_a,n_timepts_a)
#accounting for HRF
shift_a=int(5/0.693)
shift_alpha_a=np.roll(resalpha_a,shift_a)
shift_alpha_a=mne.filter.filter_data(shift_alpha_a,1/0.693,0.01,0.7)
#downsampling gamma
resgamma_a=resample(trim_gamma_a,n_timepts_a)
#accounting for HRF
shift_gamma_a=np.roll(resgamma_a,shift_a)
shift_gamma_a=mne.filter.filter_data(shift_gamma_a,1/0.693,0.01,0.7)

#Retino-1
n_timepts_r1=vol_times_r1.shape[0] #450
resalpha_r1=resample(trim_alpha_r1,n_timepts_r1)
#accounting for HRF
shift_r1=int(5/0.693)
shift_alpha_r1=np.roll(resalpha_r1,shift_r1)
shift_alpha_r1=mne.filter.filter_data(shift_alpha_r1,1/0.693,0.01,0.7)
#downsampling gamma
resgamma_r1=resample(trim_gamma_r1,n_timepts_r1)
#accounting for HRF
shift_gamma_r1=np.roll(resgamma_r1,shift_r1)
shift_gamma_r1=mne.filter.filter_data(shift_gamma_r1,1/0.693,0.01,0.7)

#Retino-2
n_timepts_r2=vol_times_r2.shape[0] #450
resalpha_r2=resample(trim_alpha_r2,n_timepts_r2)
#accounting for HRF
shift_r2=int(5/0.693)
shift_alpha_r2=np.roll(resalpha_r2,shift_r2)
shift_alpha_r2=mne.filter.filter_data(shift_alpha_r2,1/0.693,0.01,0.7)
#downsampling gamma
resgamma_r2=resample(trim_gamma_r2,n_timepts_r2)
#accounting for HRF
shift_gamma_r2=np.roll(resgamma_r2,shift_r2)
shift_gamma_r2=mne.filter.filter_data(shift_gamma_r2,1/0.693,0.01,0.7)


# In[20]:


#Step 4
#correlate with bold
                                          #ALEX-alpha
import nibabel as nib
bold_a=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_rest.nii.gz").get_fdata()
#creating an empty 4d array
alpha_4d_a=np.zeros(bold_a.shape)
alpha_4d_a=alpha_4d_a[:,:,:,0:shift_alpha_a.shape[0]]+shift_alpha_a
timepoints_a=np.arange(30,450)
mean_bold_a = np.mean(bold_a, axis=3, keepdims=True)
mean_alpha4d_a = np.mean(alpha_4d_a, axis=3, keepdims=True)
numerator_a = np.sum((bold_a[:, :, :, timepoints_a] - mean_bold_a) * (alpha_4d_a[:, :, :, timepoints_a] - mean_alpha4d_a), axis=3)
denominator_a = np.sqrt(np.sum((bold_a[:, :, :, timepoints_a] - mean_bold_a) ** 2, axis=3)) * np.sqrt(np.sum((alpha_4d_a[:, :, :, timepoints_a] - mean_alpha4d_a) ** 2, axis=3))
corrmap_alex_rest_a = numerator_a / denominator_a
fig, axes = plt.subplots(3, 11, figsize=(15, 10))  # Adjust the width and height as desired
for i in np.arange(0, 33):
    ax = axes[i // 11, i % 11]
    ax.imshow(np.rot90(corrmap_alex_rest_a[:, :, i]), vmin=-0.3, vmax=0.3)
    ax.set_title(i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the horizontal and vertical spacing
plt.show()


# In[21]:


#correlate with bold
#gamma
                                          #ALEX-gamma
import nibabel as nib
bold_a1=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_rest.nii.gz").get_fdata()

#creating an empty 4d array
gamma_4d_a=np.zeros(bold_a1.shape)
gamma_4d_a=gamma_4d_a[:,:,:,0:shift_gamma_a.shape[0]]+shift_gamma_a

timepoints_a=np.arange(30,450)

mean_bold_a = np.mean(bold_a1, axis=3, keepdims=True)
mean_gamma4d_a = np.mean(gamma_4d_a, axis=3, keepdims=True)

numerator_a1 = np.sum((bold_a1[:, :, :, timepoints_a] - mean_bold_a) * (gamma_4d_a[:, :, :, timepoints_a] - mean_gamma4d_a), axis=3)
denominator_a1 = np.sqrt(np.sum((bold_a1[:, :, :, timepoints_a] - mean_bold_a) ** 2, axis=3)) * np.sqrt(np.sum((gamma_4d_a[:, :, :, timepoints_a] - mean_gamma4d_a) ** 2, axis=3))

corrmap_alex_rest_gamma = numerator_a1 / denominator_a1
    
fig, axes = plt.subplots(3, 11, figsize=(15, 10))  # Adjust the width and height as desired

for i in np.arange(0, 33):
    ax = axes[i // 11, i % 11]
    ax.imshow(np.rot90(corrmap_alex_rest_gamma[:, :, i]), vmin=-0.3, vmax=0.3)
    ax.set_title(i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the horizontal and vertical spacing
plt.show()


# In[22]:


#Step 4
#correlate with bold
                                          #ALEX-alpha-retino-1
import nibabel as nib
bold_r1=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_01.nii.gz").get_fdata()
#creating an empty 4d array
alpha_4d_r1=np.zeros(bold_r1.shape)
alpha_4d_r1=alpha_4d_r1[:,:,:,0:shift_alpha_r1.shape[0]]+shift_alpha_r1
timepoints_r1=np.arange(30,450)
mean_bold_r1 = np.mean(bold_r1, axis=3, keepdims=True)
mean_alpha4d_r1 = np.mean(alpha_4d_r1, axis=3, keepdims=True)
numerator_r1 = np.sum((bold_r1[:, :, :, timepoints_r1] - mean_bold_r1) * (alpha_4d_r1[:, :, :, timepoints_r1] - mean_alpha4d_r1), axis=3)
denominator_r1 = np.sqrt(np.sum((bold_r1[:, :, :, timepoints_r1] - mean_bold_r1) ** 2, axis=3)) * np.sqrt(np.sum((alpha_4d_r1[:, :, :, timepoints_r1] - mean_alpha4d_r1) ** 2, axis=3))
corrmap_alex_retino_r1 = numerator_r1 / denominator_r1
fig, axes = plt.subplots(3, 11, figsize=(15, 10))  # Adjust the width and height as desired
for i in np.arange(0, 33):
    ax = axes[i // 11, i % 11]
    ax.imshow(np.rot90(corrmap_alex_retino_r1[:, :, i]), vmin=-0.3, vmax=0.3)
    ax.set_title(i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the horizontal and vertical spacing
plt.show()


# In[23]:


#correlate with bold
#gamma
                                          #ALEX-gamma-retino-1
import nibabel as nib
bold_g_r1=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_01.nii.gz").get_fdata()

#creating an empty 4d array
gamma_4d_r1=np.zeros(bold_g_r1.shape)
gamma_4d_r1=gamma_4d_r1[:,:,:,0:shift_gamma_r1.shape[0]]+shift_gamma_r1

timepoints_r1=np.arange(30,450)

mean_bold_g_r1 = np.mean(bold_g_r1, axis=3, keepdims=True)
mean_gamma4d_r1 = np.mean(gamma_4d_r1, axis=3, keepdims=True)

numerator_g_r1 = np.sum((bold_g_r1[:, :, :, timepoints_r1] - mean_bold_g_r1) * (gamma_4d_r1[:, :, :, timepoints_r1] - mean_gamma4d_r1), axis=3)
denominator_g_r1 = np.sqrt(np.sum((bold_g_r1[:, :, :, timepoints_r1] - mean_bold_g_r1) ** 2, axis=3)) * np.sqrt(np.sum((gamma_4d_r1[:, :, :, timepoints_r1] - mean_gamma4d_r1) ** 2, axis=3))

corrmap_alex_retino1_gamma = numerator_g_r1 / denominator_g_r1
    
fig, axes = plt.subplots(3, 11, figsize=(15, 10))  # Adjust the width and height as desired

for i in np.arange(0, 33):
    ax = axes[i // 11, i % 11]
    ax.imshow(np.rot90(corrmap_alex_retino1_gamma[:, :, i]), vmin=-0.3, vmax=0.3)
    ax.set_title(i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the horizontal and vertical spacing
plt.show()


# In[24]:


#Step 4
#correlate with bold
                                          #ALEX-alpha-retino-2
import nibabel as nib
bold_r2=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_02.nii.gz").get_fdata()
#creating an empty 4d array
alpha_4d_r2=np.zeros(bold_r2.shape)
alpha_4d_r2=alpha_4d_r2[:,:,:,0:shift_alpha_r2.shape[0]]+shift_alpha_r2
timepoints_r2=np.arange(30,450)
mean_bold_r2 = np.mean(bold_r2, axis=3, keepdims=True)
mean_alpha4d_r2 = np.mean(alpha_4d_r2, axis=3, keepdims=True)
numerator_r2 = np.sum((bold_r2[:, :, :, timepoints_r2] - mean_bold_r2) * (alpha_4d_r2[:, :, :, timepoints_r2] - mean_alpha4d_r2), axis=3)
denominator_r2 = np.sqrt(np.sum((bold_r2[:, :, :, timepoints_r2] - mean_bold_r2) ** 2, axis=3)) * np.sqrt(np.sum((alpha_4d_r2[:, :, :, timepoints_r2] - mean_alpha4d_r2) ** 2, axis=3))
corrmap_alex_retino2_r2 = numerator_r2 / denominator_r2
fig, axes = plt.subplots(3, 11, figsize=(15, 10))  # Adjust the width and height as desired
for i in np.arange(0, 33):
    ax = axes[i // 11, i % 11]
    ax.imshow(np.rot90(corrmap_alex_retino2_r2[:, :, i]), vmin=-0.3, vmax=0.3)
    ax.set_title(i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the horizontal and vertical spacing
plt.show()


# In[25]:


#correlate with bold
#gamma
                                          #ALEX-gamma-retino-2
import nibabel as nib
bold_g_r2=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_02.nii.gz").get_fdata()

#creating an empty 4d array
gamma_4d_r2=np.zeros(bold_g_r2.shape)
gamma_4d_r2=gamma_4d_r2[:,:,:,0:shift_gamma_r2.shape[0]]+shift_gamma_r2

timepoints_r2=np.arange(30,450)

mean_bold_g_r2 = np.mean(bold_g_r2, axis=3, keepdims=True)
mean_gamma4d_r2 = np.mean(gamma_4d_r2, axis=3, keepdims=True)

numerator_g_r2 = np.sum((bold_g_r2[:, :, :, timepoints_r2] - mean_bold_g_r2) * (gamma_4d_r2[:, :, :, timepoints_r2] - mean_gamma4d_r2), axis=3)
denominator_g_r2 = np.sqrt(np.sum((bold_g_r2[:, :, :, timepoints_r2] - mean_bold_g_r2) ** 2, axis=3)) * np.sqrt(np.sum((gamma_4d_r2[:, :, :, timepoints_r2] - mean_gamma4d_r2) ** 2, axis=3))

corrmap_alex_retino2_gamma = numerator_g_r2 / denominator_g_r2
    
fig, axes = plt.subplots(3, 11, figsize=(15, 10))  # Adjust the width and height as desired

for i in np.arange(0, 33):
    ax = axes[i // 11, i % 11]
    ax.imshow(np.rot90(corrmap_alex_retino2_gamma[:, :, i]), vmin=-0.3, vmax=0.3)
    ax.set_title(i)

plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the horizontal and vertical spacing
plt.show()


# In[26]:


import nibabel as nib
# Load the NIfTI image
mean_bold_a = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_rest.nii.gz")
# Check if mean_bold_r1 is a NIfTI image
if isinstance(mean_bold_a, nib.Nifti1Image):
    # Create a new NIfTI image with the correlation map data and the affine transformation
    cmap_a = nib.Nifti1Image(corrmap_alex_rest_a, mean_bold_a.affine, mean_bold_a.header)   
    # Save the new NIfTI image
    nib.save(cmap_a, 'C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_rest_alpha.nii.gz')
else:
    print("mean_bold_a is not a NIfTI image object.")

import nibabel as nib
# Load the NIfTI image
mean_bold_r1 = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_rest.nii.gz")
# Check if mean_bold_r1 is a NIfTI image
if isinstance(mean_bold_r1, nib.Nifti1Image):
    # Create a new NIfTI image with the correlation map data and the affine transformation
    cmap_r1 = nib.Nifti1Image(corrmap_alex_retino_r1, mean_bold_r1.affine, mean_bold_r1.header)   
    # Save the new NIfTI image
    nib.save(cmap_r1, 'C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino1_alpha.nii.gz')
else:
    print("mean_bold_r1 is not a NIfTI image object.")

mean_bold_r2=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_02.nii.gz")
cmap_r2=nib.Nifti1Image(corrmap_alex_retino2_r2,mean_bold_r2.affine,mean_bold_r2.header)
nib.save(cmap_r2,'C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino2_alpha.nii.gz')

#Gamma
mean_bold_g=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_rest.nii.gz")
cmap_g=nib.Nifti1Image(corrmap_alex_rest_gamma,mean_bold_g.affine,mean_bold_g.header)
nib.save(cmap_g,'C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_rest_gamma.nii.gz')

mean_bold_g_r1=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_01.nii.gz")
cmap_g_r1=nib.Nifti1Image(corrmap_alex_retino1_gamma,mean_bold_g_r1.affine,mean_bold_g_r1.header)
nib.save(cmap_g_r1,'C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino1_gamma.nii.gz')

mean_bold_g_r2=nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/retino_gamma_02.nii.gz")
cmap_g_r2=nib.Nifti1Image(corrmap_alex_retino2_gamma,mean_bold_g_r2.affine,mean_bold_g_r2.header)
nib.save(cmap_g_r2,'C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino2_gamma.nii.gz')


# In[27]:


t1_alex = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/t1.nii.gz").get_fdata()


# In[28]:


alex_rest_alpha=nib.load('C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_rest_alpha.nii.gz').get_fdata()
plt.subplot(231)
alex_rest_alpha.shape
plt.imshow(alex_rest_alpha[:, :,18], cmap='viridis', alpha=0.5)
plt.title('Alex-rest-alpha')
plt.axis('off')

alex_retino1_alpha=nib.load('C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino1_alpha.nii.gz').get_fdata()
plt.subplot(232)
plt.imshow(alex_retino1_alpha[:, :, 14], cmap='viridis')
plt.title('Alex-retino-1-alpha')
plt.axis('off')

alex_retino2_alpha=nib.load('C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino2_alpha.nii.gz').get_fdata()
plt.subplot(233)
plt.imshow(alex_retino2_alpha[:, :, 16], cmap='viridis')
plt.title('Alex-retino-2-alpha')
plt.axis('off')

plt.tight_layout()
plt.show()


# In[29]:


#Gamma
alex_rest_gamma=nib.load('C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_rest_gamma.nii.gz').get_fdata()
plt.subplot(231)
alex_rest_gamma.shape
plt.imshow(alex_rest_gamma[:, :,15], cmap='viridis', alpha=0.5)
plt.title('Alex-rest-gamma')
plt.axis('off')

alex_retino1_gamma=nib.load('C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino1_gamma.nii.gz').get_fdata()
plt.subplot(232)
plt.imshow(alex_retino1_gamma[:, :, 18], cmap='viridis')
plt.title('Alex-retino-1-gamma')
plt.axis('off')

alex_retino2_gamma=nib.load('C:/Users/Anjali Singh/Downloads/eegfmri/alex/corrmap_alex_retino2_gamma.nii.gz').get_fdata()
plt.subplot(233)
plt.imshow(alex_retino2_gamma[:, :, 15], cmap='viridis')
plt.title('Alex-retino-2-gamma')
plt.axis('off')

plt.tight_layout()
plt.show()


# In[30]:


average_corr_map_alpha=np.mean([corrmap_alex_rest_a,corrmap_alex_retino_r1,corrmap_alex_retino2_r2],axis=0)
average_corr_map_gamma=np.mean([corrmap_alex_rest_gamma,corrmap_alex_retino1_gamma,corrmap_alex_retino2_gamma],axis=0)

plt.subplot(221)
plt.imshow(average_corr_map_alpha[:,:,15],cmap='viridis')
plt.title("Alex-alpha-average")

plt.subplot(222)
plt.imshow(average_corr_map_gamma[:,:,17],cmap='viridis')
plt.title("Alex-gamma-average")


# In[31]:


t1_alex = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/alex/t1.nii.gz").get_fdata()
plt.imshow(t1_alex[:, :, 100], cmap='viridis')


# In[32]:


import numpy as np
from scipy.ndimage import zoom

# Assuming average_corr_map_alpha and t1_alex are defined somewhere
# average_corr_map_alpha shape: (x1, y1, z1)
# t1_alex shape: (x2, y2, z2)

# Calculate the scaling factors for each axis
scale_factors = (
    average_corr_map_alpha.shape[0] / t1_alex.shape[0],
    average_corr_map_alpha.shape[1] / t1_alex.shape[1],
    average_corr_map_alpha.shape[2] / t1_alex.shape[2]
)

# Resize the t1_alex array to match the size of average_corr_map_alpha
resized_t1_alex = zoom(t1_alex, scale_factors, order=1)
alpha=0.5
overlay_alpha = average_corr_map_alpha + alpha * resized_t1_alex

plt.imshow(np.rot90(overlay_alpha[:, :, 24]), cmap="viridis")
plt.title("Alex-alpha-overlay over T1")
plt.show() 


# In[33]:


import numpy as np
from scipy.ndimage import zoom

# Assuming average_corr_map_alpha and t1_alex are defined somewhere
# average_corr_map_alpha shape: (x1, y1, z1)
# t1_alex shape: (x2, y2, z2)

# Calculate the scaling factors for each axis
scale_factors = (
    average_corr_map_gamma.shape[0] / t1_alex.shape[0],
    average_corr_map_gamma.shape[1] / t1_alex.shape[1],
    average_corr_map_gamma.shape[2] / t1_alex.shape[2]
)

# Resize the t1_alex array to match the size of average_corr_map_alpha
resized_t1_alex = zoom(t1_alex, scale_factors, order=1)
alpha=0.5
overlay_alpha = average_corr_map_gamma + alpha * resized_t1_alex

plt.imshow(np.rot90(overlay_alpha[:, :, 20]), cmap="viridis")
plt.title("Alex-gamma-overlay over T1")
plt.show()


# In[40]:


gen_rest_alpha = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/corrmap_gen_rest_alpha.nii.gz").get_fdata()
gen_rest_gamma = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/corrmap_gen_rest_gamma.nii.gz").get_fdata()
gen_r1_alpha = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/corrmap_gen_retino1_alpha.nii.gz").get_fdata()
gen_r1_gamma = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/corrmap_gen_retino1_gamma.nii.gz").get_fdata()
gen_r2_alpha = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/corrmap_gen_retino2_alpha.nii.gz").get_fdata()
gen_r2_gamma = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/corrmap_gen_retino2_gamma.nii.gz").get_fdata()


# In[42]:


russ_rest_alpha = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/corrmap_russ_rest_alpha.nii.gz").get_fdata()
russ_rest_gamma = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/corrmap_russ_rest_gamma.nii.gz").get_fdata()
russ_r1_alpha = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/corrmap_russ_retino1_alpha.nii.gz").get_fdata()
russ_r1_gamma = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/corrmap_russ_retino1_gamma.nii.gz").get_fdata()
russ_r2_alpha = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/corrmap_russ_retino2_alpha.nii.gz").get_fdata()
russ_r2_gamma = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/corrmap_russ_retino2_gamma.nii.gz").get_fdata()


# In[43]:


t1_gen = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/genevieve/t1.nii.gz").get_fdata()
t1_russ = nib.load("C:/Users/Anjali Singh/Downloads/eegfmri/russell/t1.nii.gz").get_fdata()


# In[51]:


gen_corr_map_alpha=np.mean([gen_rest_alpha,gen_r1_alpha,gen_r2_alpha],axis=0)
gen_corr_map_gamma=np.mean([gen_rest_gamma,gen_r1_gamma,gen_r2_gamma],axis=0)

plt.subplot(221)
plt.imshow(gen_corr_map_alpha[:,:,24],cmap='viridis')
plt.title("Gen-alpha-average")

plt.subplot(222)
plt.imshow(gen_corr_map_gamma[:,:,17],cmap='viridis')
plt.title("Gen-gamma-average")


# In[53]:


russ_corr_map_alpha=np.mean([russ_rest_alpha,russ_r1_alpha,russ_r2_alpha],axis=0)
russ_corr_map_gamma=np.mean([russ_rest_gamma,russ_r1_gamma,russ_r2_gamma],axis=0)

plt.subplot(221)
plt.imshow(russ_corr_map_alpha[:,:,22],cmap='viridis')
plt.title("Russ-alpha-average")

plt.subplot(222)
plt.imshow(russ_corr_map_gamma[:,:,17],cmap='viridis')
plt.title("Russ-gamma-average")


# In[71]:


grand_avg_alpha=np.mean([russ_corr_map_alpha,gen_corr_map_alpha,average_corr_map_alpha],axis=0)
grand_avg_gamma=np.mean([russ_corr_map_gamma,gen_corr_map_gamma,average_corr_map_gamma],axis=0)

plt.subplot(221)
plt.imshow(grand_avg_alpha[:,:,22],cmap='viridis')
plt.title("Average-alpha")

plt.subplot(222)
plt.imshow(grand_avg_gamma[:,:,26],cmap='viridis')
plt.title("Average-gamma")


# In[ ]:




