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
import itertools


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

        References
        ----------
        Moon, T., and Stirling, W., Mathematical Methods and Algorithms for
        Signal Processing, Prentice Hall, 2000.
        """
        self.signal_raw = signal
        self.n = len(signal[:, 1])
        self.N = N
        self._dir = "./Voice Synth/"
        self.fname = self._dir + label + ".wav"
        self.label = label
        self.pp = 175
        self.f_s = 8000
        self.marker = itertools.cycle(('x', '+', '.', 'o', '*'))

    def _estimate_autocorrelation(self):
        """
        Estimates the autocorrelation matrix by convolving the input signal
        with a version of itself shifted by 180 degrees.
        """
        y = self.signal_raw[:, 1]  # Takes signal
        r_aa = sig.convolve(y, y[::-1])  # Convolve signal with shifted version
        self._r = r_aa[self.n - 1:self.n + self.N]  # Find appropriate auto-
        # correlation values

    def _create_YW_equations(self):
        """
        Sets up the Yule-Walker equations (an all-real variant of Eq. (8.11) in
        Moon).
        """
        self._A = np.zeros((self.N, self.N))
        self._b = np.zeros((self.N, 1))
        for i in range(self.N):
            self._A[i, i:] = self._r[:self.N-i]  # Across the rows
            self._A[i:, i] = self._r[:self.N-i]  # Down the columns
        self._b = self._r[1:]

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

    def _durbin(self):
        """
        Solves Durbin's algorithm for Toeplitz matrices. Uses `self._r` of size
        `n + 1` x `1`, where `n` is the length of the input signal minus one.
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
        self._y_imp = np.zeros(self.f_s)  # Initialize array
        i_imp = self.f_s//self.pp  # Finds the index step to place impulses
        self._y_imp[0:-1:i_imp] = 1.  # Sets impulses

    def synth_signal(self, plot=True):
        """
        The main function used in this class. Using the signal that was input
        through the class instance, as well as the number of poles in the
        all-pole filter and the label to be used in saving files, this method
        estimates the autocorrelation function to create the Yule-Walker
        equations. The equations are then solved and a periodic impulse signal
        is run through a filter with the poles found through the Yule-Walker
        equations. The output from this process is saved as `synth_sig`, which
        is the signal that is ready to be saved as a .wav file.

        See Also
        ---------
        scipy.signal.lfilter - Filters 1-D data with an IIR or FIR filter.
        scipy.signal.tf2zpk - Returns zero, pole, and gain from linear filter.
        """
        self._estimate_autocorrelation()
        self._create_YW_equations()
        self._solve_YW_equations()
        self.filter_coeffs = np.concatenate(([1.], self._a))  # Include a0
        self._generate_input_signal()
        self.synth_sig = sig.lfilter([1.], self.filter_coeffs, self._y_imp)
        [z, p, k] = sig.tf2zpk([1.], self.filter_coeffs)

        if plot:
            # Plot pole locations
            plt.figure(1)
            plt.scatter(np.real(p), np.imag(p), color='r', marker='x')
            t = np.linspace(0, np.pi*2., 100)
            plt.plot(np.cos(t), np.sin(t), color='k')
            plt.axes().set_aspect('equal')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.grid()
            plt.tight_layout()
            plt.savefig('./Figures/' + self.label + '_pole_loc.pdf')

            # Plot magnitude of transfer function
            omega = np.linspace(-np.pi, np.pi)
            sum_a = np.ones(50, dtype="complex")
            for i in range(1, self.N):
                sum_a += self.filter_coeffs[i]*np.exp(-i*(0 + omega*1j))
            H_z = np.absolute(1./(1. + sum_a))
            plt.figure(2)
            plt.plot(omega, H_z, color='k')
            plt.xlabel('$\omega$ (rad/s)')
            plt.ylabel('$|H(z)|$')
            plt.xlim((-np.pi, np.pi))
            plt.tight_layout()
            plt.savefig('./Figures/' + self.label + '_transf_mag.pdf')

    def play_sound(self):
        """
        Creates a .wav file from `synth_sig` and then plays the output signal.

        See Also
        ---------
        simpleaudio.play_buffer - Start playback of audio data.
        simpleaudio.wait_done - Waits for the playback to finish before return.
        """
        write(self.fname, 8000, self.synth_sig)  # Writes .wav file

        # Converts to appropriate form for playback from numpy array
        self.synth_sig *= 32767 / np.max(np.abs(self.synth_sig))
        self.synth_sig = self.synth_sig.astype(np.int16)

        # Audio playback
        play_obj = sa.play_buffer(self.synth_sig, 1, 2, self.f_s)
        play_obj.wait_done()

    def synthspeech(self, f, compname):
        """
        Combines multiple signals together to create a form of synthesized
        speech. The output is then saved as a .wav file and played through the
        audio output.

        Parameters
        ----------
        f : array_like
            An array containing the .wav sound files that will be combined into
            speech.
        compname : str
            A label that will be used to name the output file.

        Example
        -------
        >>> data = read("./Voice Recordings/Short _a_.wav")
        >>> y = data[1][:, 0]
        >>> test = speaknspell.SnS(np.array([x, y]).T, 'Short_a_15', 15)
        >>> test.synth_signal()
        >>> test.play_sound()

        >>> names = ['Long_a_15', 'Long_e_15', 'Long_i_15',
        >>>         'Long_o_15', 'Long_u_15']
        >>> compname = "AEIOU"
        >>> test.synthspeech(fnames, compname)

        See Also
        ---------
        simpleaudio.play_buffer - Start playback of audio data.
        simpleaudio.wait_done - Waits for the playback to finish before return.
        """
        data = read(self._dir + f[0] + ".wav")[1]  # Initializes with first
        # sound

        # Loops through remaining sound and concatenates all sounds together
        for i in f[1:]:
            data = np.concatenate((data, read(self._dir + i + ".wav")[1]))

        write(self._dir + compname + ".wav", 8000, data)  # Writes .wav file

        # Converts to appropriate form for playback from numpy array
        self.synth_phrase = data*32767 / np.max(np.abs(data))
        self.synth_phrase = self.synth_phrase.astype(np.int16)

        # Audio playback
        play_obj = sa.play_buffer(self.synth_phrase, 1, 2, self.f_s)
        play_obj.wait_done()
