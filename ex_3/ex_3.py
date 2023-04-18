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
    
    def write_wav_file(self, file_name):
        wavefile.write(file_name, self.samplerate, self.audio.astype(np.int16))

    def to_mono(self):
        # shape[0] = 44100 samples per second
        # shape[1] = 2 channels (stereo)
        if self.audio.ndim > 1 and self.audio.shape[1] > 1:
            self.audio = self.audio.mean(axis=1)
    
    def normalize(self):
        # normalized_audio = original_audio / max(abs(original_audio)) * target_level
        # convert input data to float
        audio_float = self.audio.astype(float)

        # normalize the audio
        audio_normalized = audio_float / np.max(np.abs(audio_float))

        # convert audio from float back to original format
        self.audio = (audio_normalized * np.iinfo(self.audio.dtype).max).astype(self.audio.dtype)

    def equalize(self, bands, gains):
        n = len(self.audio)
        fft_audio = np.fft.rfft(self.audio)
        freqs = np.fft.rfftfreq(n,d=1/self.samplerate)

        for band, gain in zip(bands, gains):
            freq_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))
            fft_audio[freq_idx] *= gain

        self.audio = np.fft.irfft(fft_audio)

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
   # audio_analyzer.display_waveform()
   # audio_analyzer.display_fft_spectr()

    # normalize & convert to mono
    audio_analyzer.normalize()
    audio_analyzer.to_mono()

    # perform equalizing 
    bands = [(300, 600), (600, 1200), (1200, 2400), (2400, 4800)] # frequencies
    gains = [0.0, 0.0, 1.0, 1.0]

    audio_analyzer.equalize(bands, gains)

    # save processed file
    audio_analyzer.write_wav_file("data/normalized.wav")


if __name__ == "__main__":
    main() 