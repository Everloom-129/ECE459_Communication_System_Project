import numpy as np 
import matplotlib.pyplot as plt  
from numpy.fft import fft, fftfreq, fftshift

def triangular_pulse(t, T0, amplitude=1.0):
    return np.where((t >= -T0/2) & (t <= T0/2), amplitude * (1 -(2/T0)*np.abs(t)), 0)

def generate_fft(T0,fc):
    
    fm = 0.05
    length = 10*T0   # sample range is (-length,length)
    sample_number = 1000 # need to adjust to ensure sample_freq>2*1/T0
    sample_freq = sample_number/(2*length) # related to next line
    
    t = np.linspace(-length, length, sample_number)
    #x_t = triangular_pulse(t, T0) * np.cos(2*np.pi*fc*t)
    x_t = np.cos(2*np.pi*fm*t) * np.cos(2*np.pi*fc*t)
    
    X_f = fft(x_t)   
    f = fftfreq(len(x_t), d=1 / sample_freq)  
    amp = abs(X_f)/sample_freq  # could not be abs(X_f)/N
    return t,x_t,f,amp
    


fc = 0.4
t1,x1,f1,amp1 = generate_fft(1,fc)
t2,x2,f2,amp2 = generate_fft(2,fc)
t3,x3,f3,amp3 = generate_fft(3,fc)
# draw function
plt.subplot(1, 2, 1)
plt.plot(t1, x1,label='T0 = 1')  
plt.plot(t2, x2,label='T0 = 2')  
plt.plot(t3, x3,label='T0 = 3')  
plt.title('triangular Pulse')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude')
plt.legend()  
plt.xlim(-50, 50)  #adjust to show graph clearly

    # draw fft
plt.subplot(1, 2, 2) 
plt.plot(fftshift(f1), fftshift(amp1),label='T0 = 1') 
plt.plot(fftshift(f2), fftshift(amp2),label='T0 = 2') 
plt.plot(fftshift(f3), fftshift(amp3),label='T0 = 3') 
plt.title('Fourier Transform')
plt.xlabel('frequency(Hz)') 
plt.ylabel('Magnitude')
plt.legend()
plt.xlim(-2*fc, 2*fc)  #adjust to show graph clearly


