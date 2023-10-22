# Project 2
# AM part 
# By Ruiqi Zhao
# Oct 23, 2023
import numpy as np 
import matplotlib.pyplot as plt  
from numpy.fft import fft, fftfreq, fftshift

def triangular_pulse(t, T0, amplitude=1.0):
    return np.where((t >= -T0/2) & (t <= T0/2), amplitude * (1 -(2/T0)*np.abs(t)), 0)

def generate_fft(u,fc):
    
    fm = 0.05
    length = 5   # sample range is (-length,length)
    sample_number = 2000 # need to adjust to ensure sample_freq>2*1/T0
    sample_freq = 10 # related to next line
    
    t = np.linspace(-20*length, 20*length, sample_number)
    #x_t = triangular_pulse(t, T0) * np.cos(2*np.pi*fc*t)
    x_t = (1  +  u*np.cos(2*np.pi*fm*t) ) * np.cos(2*np.pi*fc*t)
    # AM signal is (a+bcos(2*pi*fm*t)*cos(2*pi*fc*t))
    
    X_f = fft(x_t)   
    f = fftfreq(len(x_t), d=1 / sample_freq)  
    amp = abs(X_f)/sample_freq/200  # could not be abs(X_f)/N
    return t, x_t, f, amp
    


fc = 0.4
t1,x1,f1,amp1 = generate_fft(0.5,fc)
t2,x2,f2,amp2 = generate_fft(1.0,fc)
t3,x3,f3,amp3 = generate_fft(2.0,fc)
# draw function
plt.plot(t1+100, x1,label='u = 0.5')
plt.title('Time Domain')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.legend()
plt.xlim(0, 200)  #adjust to show graph clearly
plt.show()
plt.plot(t2+100, x2,label='u = 1.0')
plt.title('Time Domain')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.legend()
plt.xlim(0, 200)  #adjust to show graph clearly
plt.show()
plt.plot(t3+100, x3,label='u = 2.0')
plt.title('Time Domain')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.legend()
plt.xlim(0, 200)  #adjust to show graph clearly
plt.show()


    # draw fft
plt.plot(f1, amp1,linewidth = 0.4,label='u = 0.5')
plt.title('Frequency Domain')
plt.xlabel('frequency(Hz)')
plt.legend()
plt.show()
plt.plot(f2, amp2,linewidth = 0.4,label='u = 1.0')
plt.title('Frequency Domain')
plt.xlabel('frequency(Hz)')
plt.legend()
plt.show()
plt.plot(f3, amp3,linewidth = 0.4, label='u = 2.0')
plt.title('Frequency Domain')
plt.xlabel('frequency(Hz)')
plt.legend()
plt.show()

plt.plot(f1, amp1,label='u = 0.5')
plt.title('Frequency Domain')
plt.xlabel('frequency(Hz)')
plt.legend()
plt.xlim(0.3,0.5)
plt.ylim(0, 0.8)
plt.show()

plt.plot(f2, amp2,label='u = 1')
plt.title('Frequency Domain')
plt.xlabel('frequency(Hz)')
plt.legend()
plt.xlim(0.3,0.5)
plt.ylim(0, 0.8)
plt.show()

plt.plot(f3, amp3,label='u = 2')
plt.title('Frequency Domain')
plt.xlabel('frequency(Hz)')
plt.legend()
plt.xlim(0.3,0.5)
plt.ylim(0, 0.8)
plt.show()