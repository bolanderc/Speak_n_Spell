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
import time


class SnS:
    """
    This class represents a program that uses a linear predictor to synthesize
    recorded vowel sounds.
    """
    def __init__(self, signal, label, N):
        """
        Parameters
        ----------
        signal : array_like
            An array containing sound information from a .wav file.
        label : str
            A label that will be used to name the output file.
        N : int
            Represents the number of coefficients in the denominator of the
            transfer function H(z).

        Examples
        --------
        >>> import speaknspell
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.io.wavfile import read


        >>> data = read("./Voice Recordings/Short _a_.wav")
        >>> y = data[1][:, 0]
        >>> test = speaknspell.SnS(np.array([x, y]).T, 'Short_a_15', 15)
        >>> test.synth_signal()
        >>> test.play_sound()
        """
        self.signal_raw = signal
        self.n = len(signal[:, 1])
        self.N = N
        self._dir = "./Voice Synth/"
        self.fname = self._dir + label + ".wav"
        self.label = label
        self.pp = 175
        self.f_s = 8000

    def _estimate_autocorrelation(self):
        """

        """
        y = self.signal_raw[:, 1]
        r_aa = sig.convolve(y, y[::-1])
        self._r = r_aa[self.n - 1:self.n + self.N]

    def _create_YW_equations(self):
        """

        """
        self._A = np.zeros((self.N, self.N))
        self._b = np.zeros((self.N, 1))
        for i in range(self.N):
            self._A[i, i:] = self._r[:self.N-i]
            self._A[i:, i] = self._r[:self.N-i]
        self._b = self._r[1:]
        print(self._A)
        print(self._b)

    def _solve_YW_equations(self):
        """
        Solves the Yule-Walker equations using both numpy and a fast Toeplitz
        matrix solver called Durbin's algorithm.
        """
        t1 = time.time()
        self._a_bi = -np.linalg.solve(self._A, self._b)
        t2 = time.time()
        print(t2 - t1)
        t1 = time.time()
        self._a = -self._durbin()
        t2 = time.time()
        print(t2 - t1)
        print(self._a_bi - self._a)

    def _durbin(self):
        """
        Solves Durbin's algorithm for Toeplitz matrices. Takes `r` of size
        `n + 1` x `1` as an input, where `n` is the size of the solution `w`.
        The output of the function is `w`.
        """
        n = len(self._r) - 1
        r_o = self._r[0]

        # Solve for the initial value of w
        w = np.array([self._r[1]/r_o])

        # Loop through the rest of the matrix
        for k in range(1, n):
            A = self._r[k+1] - np.matmul(np.transpose(self._r[1:k+1]),
                                         np.flipud(w))
            B = r_o - np.matmul(np.transpose(self._r[1:k+1]), w)

            # Solve for alpha
            alpha = np.array([A/B])

            # Solve for z_k+1
            z = w - alpha*np.flipud(w)

            # Stack and output w
            w = np.array(np.concatenate((z, alpha)))
        return w

    def _generate_input_signal(self):
        """
        Generates a periodic impulse signal with a pitch period defined by the
        user with the attribute `pp`.
        """
        self._y_imp = np.zeros(self.f_s)
        i_imp = self.f_s//self.pp
        self._y_imp[0:-1:i_imp] = 1.

    def synth_signal(self, plot=True):
        """

        """
        self._estimate_autocorrelation()
        self._create_YW_equations()
        self._solve_YW_equations()
        self.filter_coeffs = np.concatenate(([1.], self._a))
        self._generate_input_signal()
        self.synth_sig = sig.lfilter([1.], self.filter_coeffs, self._y_imp)
        [z, p, k] = sig.tf2zpk([1.], self.filter_coeffs)

        # Plot pole locations
        plt.figure()
        plt.scatter(np.real(p), np.imag(p), color='r', marker='x')
        t = np.linspace(0, np.pi*2., 100)
        plt.plot(np.cos(t), np.sin(t), color='k')
        plt.axes().set_aspect('equal')
        plt.xlabel('Real')
        plt.ylabel('Imag')
        plt.grid()
        plt.savefig('./Figures/' + self.label + '_pole_loc.pdf')

        # Plot magnitude of transfer function
        omega = np.linspace(-np.pi, np.pi)
        sum_a = np.ones(50, dtype="complex")
        for i in range(1, self.N):
            sum_a += self.filter_coeffs[i]*np.exp(-i*(0 + omega*1j))
        H_z = np.absolute(1./(1. + sum_a))
        plt.figure()
        plt.plot(omega, H_z, color='k')
        plt.xlabel('$\omega$ (rad/s)')
        plt.ylabel('$|H(z)|$')
        plt.xlim((-np.pi, np.pi))
        plt.savefig('./Figures/' + self.label + '_transf_mag.pdf')

    def play_sound(self):
        """

        """
        write(self.fname, 8000, self.synth_sig)
        self.synth_sig *= 32767 / np.max(np.abs(self.synth_sig))
        self.synth_sig = self.synth_sig.astype(np.int16)
        play_obj = sa.play_buffer(self.synth_sig, 1, 2, self.f_s)
        play_obj.wait_done()

    def synthspeech(self, f, compname):
        """

        """
        data = read(self._dir + f[0] + ".wav")[1]
        for i in f[1:]:
            data = np.concatenate((data, read(self._dir + i + ".wav")[1]))
        write(self._dir + compname + ".wav", 8000, data)
