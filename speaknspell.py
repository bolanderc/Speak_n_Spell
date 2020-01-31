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


class SnS:
    def __init__(self, signal, N):
        self.signal_raw = signal
        self.n = len(signal[:, 1])
        self.N = N

    def _estimate_autocorrelation(self):
        y = self.signal_raw[:, 1]
        r_aa = sig.convolve(y, y[::-1])
        print(self.n + self.N -1)
        self._r = r_aa[self.n - 1:self.n + self.N]
        print(self._r)

    def _create_YW_equations(self):
        self._A = np.zeros((self.N, self.N))
        self._b = np.zeros((self.N, 1))
        for i in range(self.N):
            self._A[i, i:] = self._r[:self.N-i]
            self._A[i:, i] = self._r[:self.N-i]
        print(self._A)
        self._b = self._r[1:]
        print(self._b)

    def _solve_YW_equations(self):
        self._a = -np.linalg.solve(self._A, self._b)
        print(self._a)

    def _generate_input_signal(self):
        pass

    def synth_signal(self):
        self._estimate_autocorrelation()
        self._create_YW_equations()
        self._solve_YW_equations()
        self.filter_coeffs = np.concatenate(([1.], self._a))
        print(self.filter_coeffs)

    def play_sound(self):
        pass

x = np.linspace(0, 3, num=1500)
y = np.sin(50*x)
plt.plot(x, y)
test = SnS(np.array([x, y]).T, 10)
test.synth_signal()
yt = test._a
