# DSB-SC Realization 

Graph representation

Property analysis



## experimental setup

Same as

The experiment focuses on the study of sinusoidal modulation based on the setup of section  3.1 Amplitude Modulation (AM). The parameters are set as follows:

- **Carrier amplitude: **$ A_c =  1$.
- **Carrier frequency:**  $ f_c =  0.4 Hz$.
- **Modulation frequency:** $ f_m =  0.05 Hz$.

For the purpose of visualization and analysis, the experiment aims to showcase 10 complete cycles of the modulated wave, which corresponds to a total time span of 200 seconds. To achieve this on a digital system, the modulated wave is sampled at a rate of 10 Hz (fs = 10 Hz). This sampling rate results in a total of 2,000 data points for the entire duration (200 seconds multiplied by the sampling rate of 10 Hz).

The frequency spectrum of the modulated signal occupies a bandwidth that stretches from -5 Hz to 5 Hz. Given the proximity of the carrier frequency to its adjacent side frequencies (a separation of 0.05 Hz, which is the modulation frequency), there is a desire to achieve a fine frequency resolution of 0.005 Hz (fr). This ensures accurate representation and differentiation between the carrier and the side frequencies.



DSB-SC modulation: 

### (a) DSB-SC modulated wave, 

### (b) magnitude spectrum of the modulated wave,



### (c) expanded spectrum around the carrier frequency

I apologize for the oversight. Let's break down the provided graphs to understand them more clearly and guide our code implementation.

### Observations from the Graphs:

**Graph (a)** - Waveform of signal produced at the output of product modulator:
- This waveform is the result of multiplying the DSB-SC modulated signal with the carrier. 
- We can observe that there are two main frequencies in the waveform. The lower frequency is the original message frequency (fm) and the higher frequency is twice the carrier frequency (2fc).

**Graph (b)** - Amplitude spectrum of the signal in part (a):
- We can observe two distinct peaks.
- The spectrum shows frequency components at the baseband (which corresponds to the original message) and at twice the carrier frequency.

**Graph (c)** - Waveform of low-pass filter output:
- After passing through the low-pass filter, the higher frequency component (2fc) is removed.
- The waveform appears as the original message signal.

**Graph (d)** - Amplitude spectrum of signal in part (c):
- Only the baseband frequency is retained, which corresponds to the original message frequency.

### Steps to Generate the Correct Graphs:

1. **Produce the Signal at the Product Modulator's Output**:
    - Multiply the DSB-SC signal by the carrier.
    - The result will have the original message frequency and a frequency at twice the carrier frequency.

2. **Generate the Amplitude Spectrum for the Signal from Step 1**:
    - Perform an FFT on the signal from step 1.
    - Plot the magnitude of the FFT result.

3. **Low-Pass Filter the Signal from Step 1**:
    - Use a low-pass filter to remove the higher frequency component (2fc).
    - The output will be similar to the original message signal.

4. **Generate the Amplitude Spectrum for the Signal from Step 3**:
    - Perform an FFT on the filtered signal.
    - Plot the magnitude of the FFT result.

Given these steps and observations, let's re-implement the code.

Alright, I'll provide a step-by-step breakdown, without writing the full code immediately.

### 1. Waveform of signal produced at the output of product modulator:

Given the DSB-SC modulated wave, multiplying it with an exact replica of the carrier will result in two primary components:

(a) The baseband message signal \( m(t) \), which represents the original information.
(b) A component at twice the carrier frequency \( 2f_c \). 

Mathematically, this operation can be represented as:
\[ x(t) = m(t) \times \cos(2\pi f_c t) \times \cos(2\pi f_c t) \]

This multiplication results in the sum and difference frequencies. Given that DSB-SC has suppressed the carrier, we only have the original message and the frequency component at twice the carrier frequency.

### 2. Amplitude spectrum of the signal in part (a):

Upon plotting the frequency domain representation of the above waveform using the FFT, you will observe two primary spectral components:

(a) The baseband message spectrum.
(b) A spectrum centered around twice the carrier frequency \( 2f_c \).

### 3. Waveform of low-pass filter output:

Passing the product signal (from step 1) through a low-pass filter aims to eliminate the component at \( 2f_c \) and retain only the baseband message signal. So, the resulting waveform will look similar to the original message signal, \( m(t) \), which is sinusoidal.

### 4. Amplitude spectrum of signal in part (c):

The frequency domain representation of the low-pass filter output will show only the baseband message spectrum, which is just a single peak (as depicted in the graph you provided).

---

Given the above steps, I understand the workflow and the expected outcomes more clearly now. Would you like me to proceed with coding this step by step?





Certainly! The power of a signal is an important metric in communications, as it can affect transmission range, quality of reception, and energy efficiency.

### Power of DSB-SC (Double Sideband Suppressed Carrier):

For a DSB-SC modulated signal, the modulating message signal \( m(t) \) multiplies with a carrier wave to create the DSB-SC signal, but the carrier itself doesn't appear in the transmitted signal, hence "suppressed carrier".

Given:
\[ s(t) = m(t) \cos(2\pi f_c t) \]

Where:
- \( s(t) \): DSB-SC signal
- \( m(t) \): Message signal
- \( f_c \): Frequency of the carrier

The power \( P_{DSB-SC} \) of the DSB-SC signal \( s(t) \) is given by:
\[ P_{DSB-SC} = \frac{1}{T} \int_0^T |s(t)|^2 dt \]

For the DSB-SC signal, this power becomes:
\[ P_{DSB-SC} = \frac{1}{T} \int_0^T m^2(t) \cos^2(2\pi f_c t) dt \]

Using the trigonometric identity:
\[ \cos^2(\theta) = \frac{1 + \cos(2\theta)}{2} \]

The power becomes:
\[ P_{DSB-SC} = \frac{1}{T} \int_0^T m^2(t) \frac{1 + \cos(4\pi f_c t)}{2} dt \]

Since the double-frequency term averages out to zero over a period, the power of the DSB-SC signal simplifies to:
\[ P_{DSB-SC} = \frac{1}{2T} \int_0^T m^2(t) dt \]

This shows that the power of the DSB-SC signal is half the power of the modulating message signal (assuming the carrier has a peak amplitude of 1).

**Conclusion:**
The power in the DSB-SC signal is solely due to the message signal since the carrier does not contribute any power (as it is suppressed). Thus, the energy efficiency of DSB-SC is generally higher than standard AM with a carrier, but at the cost of requiring coherent detection for demodulation.



