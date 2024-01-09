
 # SSB.py - main source code for ECE459 Project 4 
 #
 # Author:	    Wang Suhao
 # Version:	    1
 # Creation Date:   Thu Nov 23 12:34:30 2023
 # Filename:	    SSB.py
 # History:
 #	SL	1	Thu Nov 23 12:34:30 2023
 #		First written.


import numpy as np 
import matplotlib.pyplot as plt  
from numpy.fft import fft, fftfreq, ifft

# calculate bandwidth, return bandwidth
def cal_bandwidth(amp,f,fc):
    j = 0
    maximum = 0
    for j in range(len(f)):
        if amp[j]>maximum:
            maximum = amp[j]
    print(maximum)
    i = 0
    if (fc > 0):
        while amp[i]<=0.001*maximum:
            i += 1
        result = 2*(fc-f[i])
    if (fc == 0):
        while amp[i]>0.001*maximum:
            i += 1
        result = 2*(f[i]-fc)
    return result

# generate pulse in (T1,T1+dt) with amplitude
def square_pulse(t,T1,dt,amplitude = 1.0):
    return np.where((t >= T1-dt) & (t <= T1+dt), amplitude, 0)

def triangular_pulse(t, T0, amplitude=1.0):
    return np.where((t >= -T0/2) & (t <= T0/2), amplitude * (1 -(2/T0)*np.abs(t)), 0)

# generate mask
def generate_mask( f, fc, f_low , f_high):
    mask1 = np.logical_and(f >= f_low, f <= f_high)
    mask2 = np.logical_and(f >= -f_high, f <= -f_low)
    mask = np.logical_or(mask1,mask2)
    if (fc == 0):
        return mask1
    else:
        return mask

    
##############################################
###  第一个信号
##############################################


duration =  0.1                 
center = 0
fc = 50

fs = 500 
t = np.linspace(-100, 100, 200*fs)
                  

# origin x1
x1 = square_pulse(t, center, duration)*np.cos(2*np.pi*fc*t)
f1 = fftfreq(len(x1), d=1 / fs)
X_f1 = fft(x1)/fs    #非周期函数归一化要乘以1/fs


# get ussb of x1
bandwidth1 = cal_bandwidth(abs(X_f1),f1,fc) # 参数是amp而非X
mask1 = generate_mask(f1,fc,fc,fc+bandwidth1/2)

f1_ussb = f1
X_f1_ussb = X_f1*mask1
x1_ussb = ifft(X_f1_ussb)*2*fs  #我也不知道为什么要乘以2fs，ifft也要归一化吗

##############################################
### 第二个信号
##############################################         
center = 0
Band = 30
amplitude = 10
                  

# origin x2
x2 = triangular_pulse(t, 0.5)*np.cos(2*np.pi*fc*t)
f2 = fftfreq(len(x2), d=1 / fs)
X_f2 = fft(x2)/fs    #非周期函数归一化要乘以1/fs

# get lssb of x2
bandwidth2 = cal_bandwidth(abs(X_f2),f2,fc)  # 参数是amp而非X
mask2 = generate_mask(f1,fc,fc-bandwidth2/2,fc)

f2_lssb = f2
X_f2_lssb = X_f2*mask2
x2_lssb = ifft(X_f2_lssb)*2*fs  #我也不知道为什么要乘以2fs，ifft也要归一化吗


##############################################
##############################################
f_combine = f1
Y_combine = X_f2_lssb+X_f1_ussb
y_combine = ifft(Y_combine) *2*fs #我也不知道为什么要乘以2fs，ifft也要归一化吗


##############################################
#####        demodulation
##############################################

### 第一次滤波 ###
# LPF的Bc意思是得到[-50,50],h = 2Bcsinc(2Bct)
Bc = 50            
sample_center = 0  # 从-100取样到100，取样中心为0
h_lp = 2*Bc*np.sinc(2*Bc*(t-sample_center))
y_lssb = np.convolve(y_combine,h_lp,'same')/fs  #归一化

# fft还是要再次归一化
Y_lssb = fft(y_lssb)/fs


# BPF的Bc意思是得到[f1-Bc/2,f1+Bc/2],h = 2Bcsinc(Bct)*cos(2pi*f1*t)
Bc2 = 50
h_hp = 2*Bc2*np.sinc(Bc2*(t-sample_center))*np.cos(2*np.pi*75*t)
y_ussb = np.convolve(y_combine,h_hp,'same')/fs

#如果m1和m2,原函数的中心在0，则fft还是要再次归一化
Y_ussb = fft(y_ussb)/fs


### 乘以cos再乘以2 ###
m1 = y_lssb*np.cos(2*np.pi*fc*t)*2
m2 = y_ussb*np.cos(2*np.pi*fc*t)*2
M1 = fft(m1)/fs
M2 = fft(m2)/fs


### 再次滤波 ###
m3 = np.convolve(m1,h_lp,'same')/fs
m4 = np.convolve(m2,h_lp,'same')/fs
M3 = fft(m3)/fs
M4 = fft(m4)/fs



#########################################################
######   combine siganl
#########################################################
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# draw x1
plt.subplot(4, 4, 1)
plt.plot(t, x1)  
plt.title('Time-Domain Signal x1(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly


# draw fft of x1
plt.subplot(4, 4, 2) 
plt.plot(f1, abs(X_f1)) 
plt.title('Frequency Spectrum of x1(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly

# draw x2
plt.subplot(4, 4, 3)
plt.plot(t, x2)  
plt.title('Time-Domain Signal x2(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly


# draw fft of x2
plt.subplot(4, 4, 4) 
plt.plot(f2, abs(X_f2)) 
plt.title('Frequency Spectrum of x2(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly


# draw ussb
plt.subplot(4, 4, 5)
plt.plot(t, x1_ussb)  
plt.title('Time-Domain Signal ussb(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw fft of ussb
plt.subplot(4, 4, 6) 
plt.plot(f1_ussb, abs(X_f1_ussb)) 
plt.title('Frequency Spectrum of ussb(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly

# draw lssb
plt.subplot(4, 4, 7)
plt.plot(t, x2_lssb)  
plt.title('Time-Domain Signal lssb(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw fft of lssb
plt.subplot(4, 4, 8) 
plt.plot(f2_lssb, abs(X_f2_lssb)) 
plt.title('Frequency Spectrum of lssb(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly


#########################################################
######   combine siganl
#########################################################
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1) 
# draw y_combine
plt.plot(t, y_combine)  
plt.title('Time-Domain Signal y(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw fft of y_combine
plt.subplot(1, 2, 2) 
plt.plot(f_combine, abs(Y_combine)) 
plt.title('Frequency Spectrum of y(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly


#########################################################
###### demodulation
#########################################################
plt.figure(figsize=(15, 15))
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# draw y_lssb
plt.subplot(4, 4, 1)
plt.plot(t, y_lssb)  
plt.title('Time-Domain Signal y_lssb(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw fft of y_lssb
plt.subplot(4, 4, 2) 
plt.plot(f_combine, abs(Y_lssb)) 
plt.title('Frequency Spectrum of y_lssb(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly

# draw y_ussb
plt.subplot(4, 4, 3)
plt.plot(t, y_ussb)  
plt.title('Time-Domain Signal y_ussb(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw fft of y_ussb
plt.subplot(4, 4, 4) 
plt.plot(f_combine, abs(Y_ussb)) 
plt.title('Frequency Spectrum of y_ussb(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly


# draw original m1 without lpf
plt.subplot(4, 4, 5)
plt.plot(t, m1)  
plt.title('Time-Domain Signal m1_without_lpf(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw original m1 without lpf
plt.subplot(4, 4, 6)
plt.plot(f_combine, abs(M1))  
plt.title('Frequency Spectrum of m1_without_lpf(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly


# draw orginal m2 without lpf
plt.subplot(4, 4, 7) 
plt.plot(t, m2) 
plt.title('Time-Domain Signal m2_without_lpf(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly


# draw orginal m2 without lpf
plt.subplot(4, 4, 8) 
plt.plot(f_combine, abs(M2)) 
plt.title('Frequency Spectrum of m2_without_lpf(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly



# draw orginal m1 
plt.subplot(4, 4, 9) 
plt.plot(t, m3) 
plt.title('Time-Domain Signal m1(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly


# draw orginal m1
plt.subplot(4, 4, 10) 
plt.plot(f_combine, abs(M3)) 
plt.title('Frequency Spectrum of m1(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly

# draw orginal m2 without lpf
plt.subplot(4, 4, 11) 
plt.plot(t, m4) 
plt.title('Time-Domain Signal m2(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(center-1,center+1)  #adjust to show graph clearly

# draw orginal m2
plt.subplot(4, 4, 12) 
plt.plot(f_combine, abs(M4)) 
plt.title('Frequency Spectrum of m2(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly

plt.show()


