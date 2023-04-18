import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavefile 

from scipy import signal
from scipy.signal import butter, filtfilt

class AudioAnalyzer: 
    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.length, self.samplerate, self.audio = self.load_wav_file(file_name)

    def load_wav_file(self, file_name):
        samplerate, audio = wavefile.read(file_name)
        length = audio.shape[0] / samplerate
        return length, samplerate, audio
    
    def display_info(self):
        print(f"Number of channels: { self.audio.shape[1] }")
        print(f"Sample rate: { self.samplerate}")

    def display_waveform(self):
        # retrive time in seconds
        time = np.linspace(0., self.length, self.audio.shape[0])

        plt.plot(time, self.audio[:, 0], label="Left Channel")
        plt.plot(time, self.audio[:, 1], label = "Right Channel")

        plt.legend()

        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.show()

    def display_fft_spectr(self):
        plt.figure()
        plt.specgram(self.audio[:, 0], NFFT=1024, Fs=self.samplerate, noverlap=900 )
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar()
        plt.show()

def main():
    audio_analyzer = AudioAnalyzer("data/example.wav")
    audio_analyzer.display_info()
    audio_analyzer.display_waveform()
    audio_analyzer.display_fft_spectr()

if __name__ == "__main__":
    main() 