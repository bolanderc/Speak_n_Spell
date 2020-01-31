#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:35:40 2020

@author: christian
"""

import numpy as np
import scipy.signal as sig
import simpleaudio as sa
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


class SnS:
    def __init__(self, signal, N):
        self.signal_raw = signal
        self.n = len(signal[:, 1])
        self.N = N

    def _estimate_autocorrelation(self):
        y = self.signal_raw[:, 1]
        r_aa = sig.convolve(y, y[::-1])
        self._r = r_aa[self.n - 1:self.n + self.N]
        plt.figure(1)
        plt.plot(self.signal_raw[:, 0], r_aa[:self.n])

    def _create_YW_equations(self):
        self._A = np.zeros((self.N, self.N))
        self._b = np.zeros((self.N, 1))
        for i in range(self.N):
            self._A[i, i:] = self._r[:self.N-i]
            self._A[i:, i] = self._r[:self.N-i]
        self._b = self._r[1:]

    def _solve_YW_equations(self):
        self._a = -np.linalg.solve(self._A, self._b)

    def _generate_input_signal(self):
        f_s = 8000
        pp = 175
        self._y_imp = np.zeros(f_s)
        i_imp = f_s//pp
        self._y_imp[0:-1:i_imp] = 1.

    def synth_signal(self):
        self._estimate_autocorrelation()
        self._create_YW_equations()
        self._solve_YW_equations()
        self.filter_coeffs = np.concatenate(([1.], self._a))
        print(self.filter_coeffs)
        self._generate_input_signal()
        self.synth_sig = sig.lfilter([1.], self.filter_coeffs, self._y_imp)

    def play_sound(self):
        plt.figure(10)
        plt.plot(self.synth_sig)


#wave_obj = sa.WaveObject.from_wave_file("./Voice Recordings/_Ah_.wav")
#play_obj = wave_obj.play()
#play_obj.wait_done()
data = read("./Voice Recordings/_Ah_.wav")
y = data[1][:, 0]
x = np.arange(0, len(y), 1)
#x = np.linspace(0, 3, num=1500)
#y = np.sin(50*x)
plt.figure(0)
plt.plot(x, y)
test = SnS(np.array([x, y]).T, 10)
test.synth_signal()
yt = test._a
test.play_sound()
