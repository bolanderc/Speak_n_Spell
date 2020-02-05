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
from scipy.io.wavfile import read, write


class SnS:
    def __init__(self, signal, label, N):
        self.signal_raw = signal
        self.n = len(signal[:, 1])
        self.N = N
        self._dir = "./Voice Synth/"
        self.fname = self._dir + label + ".wav"

    def _estimate_autocorrelation(self):
        y = self.signal_raw[:, 1]
        r_aa = sig.convolve(y, y[::-1])
        self._r = r_aa[self.n - 1:self.n + self.N]

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

    def synth_signal(self, plot=True):
        self._estimate_autocorrelation()
        self._create_YW_equations()
        self._solve_YW_equations()
        self.filter_coeffs = np.concatenate(([1.], self._a))
        self._generate_input_signal()
        self.synth_sig = sig.lfilter([1.], self.filter_coeffs, self._y_imp)
        [z, p, k] = sig.tf2zpk([1.], self.filter_coeffs)

        # Plot pole locations
        plt.figure()
        plt.scatter(np.real(p), np.imag(p))
        t = np.linspace(0, np.pi*2., 100)
        plt.plot(np.cos(t), np.sin(t))
        plt.axes().set_aspect('equal')

        # Plot magnitude of transfer function
        omega = np.linspace(-np.pi, np.pi)
        sum_a = np.ones(50, dtype="complex")
        for i in range(1, self.N):
            sum_a += self.filter_coeffs[i]*np.exp(-i*(0 + omega*1j))
        H_z = np.absolute(1./(1. + sum_a))
        plt.figure()
        plt.plot(omega, H_z)

    def play_sound(self):
        write(self.fname, 8000, self.synth_sig)
        wave_obj = sa.WaveObject.from_wave_file(self.fname)
        play_obj = wave_obj.play()
        play_obj.wait_done()

    def synthspeech(self, f, compname):
        data = read(self._dir + f[0] + ".wav")[1]
        j = 0
        for i in f[1:]:
            data = np.concatenate((data, read(self._dir + i + ".wav")[1]))
        write(self._dir + compname + ".wav", 8000, data)



data = read("./Voice Recordings/Short _a_.wav")
y = data[1][:, 0]
x = np.arange(0, len(y), 1)
plt.figure(0)
plt.plot(x, y)
test = SnS(np.array([x, y]).T, 'Short_a_15', 15)
test.synth_signal()
test.play_sound()

#fnames = ['Long_a_15', 'Long_e_15', 'Long_i_15',
#          'Long_o_15', 'Long_u_15']
#compname = "AEIOU"
#test.synthspeech(fnames, compname)
