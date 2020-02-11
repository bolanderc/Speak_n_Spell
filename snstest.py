#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:35:30 2020

@author: christian
"""


import speaknspell
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
plt.style.use('paper')


data = read("./Voice Recordings/Short _a_.wav")
y = data[1][:, 0]
x = np.arange(0, len(y), 1)
#    plt.figure(0)
#    plt.plot(x, y)
test = speaknspell.SnS(np.array([x, y]).T, "Short _a_"[:-1], 15)
test.synth_signal()
test.play_sound()

#filenames = ["Short _a_", "Short _e_", "Short _i_", "Short _o_",
#             "Short _u_", "Long _a_", "Long _e_", "Long _i_",
#             "Long _o_", "Long _u_", "_Ah_"]
#for filename in filenames:
#    data = read("./Voice Recordings/" + filename + ".wav")
#    y = data[1][:, 0]
#    x = np.arange(0, len(y), 1)
##    plt.figure(0)
##    plt.plot(x, y)
#    test = speaknspell.SnS(np.array([x, y]).T, filename[:-1], 15)
#    test.synth_signal()
#    test.play_sound()





#fnames = ['Long_a_15', 'Long_e_15', 'Long_i_15',
#          'Long_o_15', 'Long_u_15']
#compname = "AEIOU"
#test.synthspeech(fnames, compname)
