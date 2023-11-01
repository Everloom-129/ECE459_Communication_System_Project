
 # BPF.py - main source code for ECE459 Project 3 
 #
 # Author:	    Wang Suhao
 # Version:	    1
 # Creation Date:   Wed Oct 25 19:56:30 2023
 # Filename:	    BPF.py
 # History:
 #	SL	1	Thu Oct 26 10:09:30 2023
 #		First written.
 #	SL	2	Fri Oct 27 09:38:18 2023
 #		rewrite and test periodic pulses
 #	SL	3	Fri Oct 27 14:38:18 2023
 #		rewrite and polish for project3

import numpy as np 
import matplotlib.pyplot as plt  
from numpy.fft import fft, fftfreq, fftshift, ifft
import math

# calculate 3db bandwidth point
def three_db_bandwidth(amp,f,fc):
    j = 0
    maximum = 0
    for j in range(len(f)):
        if amp[j]>maximum:
            maximum = amp[j]
    print(maximum)
    i = 0
    if (fc > 0):
        while amp[i]<=math.sqrt(2)*maximum/2:
            i += 1
        result = 2*(fc-f[i])
    if (fc == 0):
        while amp[i]>math.sqrt(2)*maximum/2:
            i += 1
        result = 2*(f[i]-fc)
    return result

# generate pulse in (T1,T1+dt) with amplitude
def square_pulse(t,T1,dt,amplitude = 1.0):
    return np.where((t >= T1-dt) & (t <= T1+dt), amplitude, 0)


# generate fft with ideal BPF or not
def generate_fft(sig, BPF = 0, f_low = 0, f_high = 0):
    f = fftfreq(len(sig), d=1 / fs)  
    X_f = fft(sig)
    amp = abs(X_f)/len(sig) # is not abs(X_f)/N

    if (BPF == 1):
        mask1 = np.logical_and(f >= f_low, f <= f_high)
        mask2 = np.logical_and(f >= -f_high, f <= -f_low)
        mask = np.logical_or(mask1,mask2)
        X_f = X_f * mask
        amp = abs(X_f)/len(sig)
        
    return f, X_f, amp

# generate function x_t which combines pulses
def generate_function(fs, fc, dt, fm, am):
    # check if fs>2fm
    length = 100
    N = 2*length*fs
    
    t = np.linspace(-length, length, N)
    x_t = 0
    for i in range(0,len(fm)):
        x_t += square_pulse(t, fm[i], dt, am[i])
    
    x_t = x_t * np.cos(2*np.pi*fc*t)
    return t,x_t

# generate perodic function x_t which combines square_pulses
def periodic_function(fs, fc, dt, T0):
    # check if fs>2fm
    length = 10
    N = 2*length*fs
    
    t = np.linspace(-length, length, N)
    x_t = 0
    for i in range(-length,length,T0):
        x_t += square_pulse(t, i, dt, 1)
    
    x_t = x_t * np.cos(2*np.pi*fc*t)
    return t,x_t
    


# fm: the left position of pulse
# am: the amplitude of pulse
# dt: the duration of pulse
# fs: the sample frequency
# low: the BPF low frequency
# high:  the BPF high frequency

fm = [5, 10, 15, 20, 30]    
am = [50, 0, 0, 0, 0]  
duration =  0.1                 
fs = 1000 
fc = 0
                  

# origin signal and fft
t1, x1 = generate_function(fs, fc, duration, fm, am)
#t1, x1 = periodic_function(fs, fc, duration, 1)
f1, X_f1, amp1 = generate_fft(x1)
# period = 5
# t1, x1 = periodic_function(fs, fc, duration, period)

# calculate 3db bandwidth point
# 0.8 and 8
Bc_is = 8;
Bc_is = 0.8;
bandwidth = three_db_bandwidth(amp1,f1,fc)
print('B is %d',bandwidth)
low =  fc - Bc_is * bandwidth/2                    
high = fc + Bc_is * bandwidth/2
print('Bc is %d',Bc_is*bandwidth)

# signal and fft after BPF
f2, X_f2, amp2 = generate_fft(x1, 1, low, high)
x2 = ifft(X_f2)


plt.figure(figsize=(9, 6))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# draw function
plt.subplot(2, 2, 1)
plt.plot(t1, x1)  
plt.title('Time-Domain Signal x(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(4, 6)  #adjust to show graph clearly


# draw fft
plt.subplot(2, 2, 2) 
plt.plot(fftshift(f1), fftshift(amp1)) 
plt.title('Frequency Spectrum of x(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly



# draw function afer BPF
plt.subplot(2, 2, 3)
plt.plot(t1, x2)  
plt.title('Time-Domain Signal y(t)')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude') 
plt.grid(True)
#plt.legend()
plt.xlim(4, 6)  #adjust to show graph clearly

# draw fft after BPF
plt.subplot(2, 2, 4) 
plt.plot(fftshift(f2), fftshift(amp2)) 
plt.title('Frequency Spectrum of y(t)')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.grid(True)
#plt.legend()
plt.xlim(-100,100)  #adjust to show graph clearly
