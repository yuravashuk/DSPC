import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavefile 

from scipy import signal
from audio_processor import AudioProcessor

# Provide info about audio signal (waveform, fft spectre, file data)
class AudioInfoProvider:
    def __init__(self, audio_processor) -> None:
        self.audio_processor = audio_processor

    def display_info(self):
        print(f"Number of channels: { self.audio_processor.audio.shape[1] }")
        print(f"Sample rate: { self.audio_processor.samplerate}")

    def display_waveform(self):
        # retrive time in seconds
        time = np.linspace(0., self.audio_processor.length, self.audio_processor.audio.shape[0])

        plt.plot(time, self.audio_processor.audio[:, 0], label = "Left Channel")
        plt.plot(time, self.audio_processor.audio[:, 1], label = "Right Channel")

        plt.legend()

        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.show()

    def display_fft_spectr(self):
        plt.figure()
        plt.specgram(self.audio_processor.audio[:, 0], NFFT=1024, Fs=self.audio_processor.samplerate, noverlap=900 )
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar()
        plt.show()