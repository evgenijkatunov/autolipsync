import numpy.fft as fft
import numpy as np
import math


class Frame:
    _start_frequency = 20.
    _end_frequency = 20000.
    _mel_count = 15
    wave = []
    start_time = 0
    duration = 0
    framerate = 0
    spectre = None
    frequencies = None
    phonemes = []
    phoneme = None

    def __init__(self, wave, start_time, duration, framerate, phonemes):
        self.wave = wave
        self.start_time = start_time
        self.duration = duration
        self.framerate = framerate
        self.phonemes = phonemes

    def fft(self):
        self.spectre = fft.rfft(self.wave)
        self.frequencies = fft.rfftfreq(len(self.wave), 1. / self.framerate)
        # handle_line, = plt.plot(self.frequences, abs(self.spectre), label=self.start_time)
        # plt.legend(handles=[handle_line])
        # plt.show()

    def mel(self):
        start_mel = 1127 * math.log(1 + self._start_frequency / 700)
        end_mel = 1127 * math.log(1 + self._end_frequency / 700)
        mel_frequencies = []
        for i in range(0, self._mel_count + 2):
            mel_frequencies.append(start_mel + (end_mel - start_mel) * float(i) / (self._mel_count + 1))
        frequencies = [700 * (math.exp(frequency / 1127) - 1) for frequency in mel_frequencies]
        windows = []
        for i in range(0, self._mel_count):
            windows.append([frequencies[i], frequencies[i + 1], frequencies[i + 2]])
            #handle_line, = plt.plot([frequencies[i], frequencies[i + 1], frequencies[i + 2]], [0, 1, 0], label=self.start_time)
            #plt.legend(handles=[handle_line])

        #plt.show()
        energy = np.zeros(self._mel_count)
        window_number = 0
        for i in range(0, len(self.spectre)):
            frequency = self.frequencies[i]
            if frequency > windows[window_number][2]:
                window_number += 1
                if window_number >= self._mel_count:
                    break
            window = windows[window_number]
            if frequency > window[0]:
                if frequency > window[1]:
                    energy[window_number] += math.pow(abs(self.spectre[i]), 2) * (
                        float(window[2] - frequency) / (window[2] - window[1]))
                else:
                    energy[window_number] += math.pow(abs(self.spectre[i]), 2) * (
                        float(frequency - window[0]) / (window[1] - window[0]))
            if window_number < self._mel_count - 1:
                window = windows[window_number+1]
                if frequency > window[0]:
                    if frequency > window[1]:
                        energy[window_number+1] += math.pow(abs(self.spectre[i]), 2) * (
                            float(window[2] - frequency) / (window[2] - window[1]))
                    else:
                        energy[window_number+1] += math.pow(abs(self.spectre[i]), 2) * (
                            float(frequency - window[0]) / (window[1] - window[0]))
        energy = list(map(lambda x: math.log(x) if x > 0 else x, energy))
        self.mfcc = np.zeros((len(energy)))
        for i in range(0, len(energy)):
            mfcc = 0
            for j in range(0, len(energy)):
                mfcc += energy[j]*math.cos(math.pi*i*(j + 0.5)/len(energy))
            self.mfcc[i] = mfcc
        #handle_line, = plt.plot(self.phonemes)
        #plt.legend(handles=[handle_line])
        #plt.show()

