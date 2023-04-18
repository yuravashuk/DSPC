import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavefile 

from scipy import signal

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


# Load, process and save audio files
class AudioProcessor: 
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

    def equalize(self, bands, gains_db):
        n = len(self.audio)
        fft_audio = np.fft.rfft(self.audio)
        freqs = np.fft.rfftfreq(n,d=1/self.samplerate)

        for band, gain in zip(bands, gains_db):
            freq_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))
            linear_gain = 10 ** (gain / 20)
            fft_audio[freq_idx] *= linear_gain

        self.audio = np.fft.irfft(fft_audio)

    def stft(self, x, fft_size, hop_size):
        window = np.hanning(fft_size)
        return np.array([np.fft.rfft(window * x[i:i + fft_size]) for i in range(0, len(x) - fft_size, hop_size)])

    def istft(self, X, fft_size, hop_size):
        window = np.hanning(fft_size)
        x = np.zeros((X.shape[0] - 1) * hop_size + fft_size)
        for n, frame in enumerate(X):
            x[n * hop_size:n * hop_size + fft_size] += np.real(np.fft.irfft(frame)) * window
        return x

    def spectral_subtraction(self, alpha=2.0, beta=0.15):
        # Set FFT size and hop size
        fft_size = 512
        hop_size = int(fft_size / 2)

        # Compute the Short-Time Fourier Transform (STFT)
        stft_data = self.stft(self.audio, fft_size, hop_size)

        # Estimate the noise spectrum by averaging the magnitudes of the first few frames
        noise_spectrum = np.mean(np.abs(stft_data[:5]), axis=0)

        # Perform spectral subtraction
        de_noised_spectrum = np.abs(stft_data) - alpha * noise_spectrum[np.newaxis, :]
        de_noised_spectrum = np.maximum(de_noised_spectrum, beta * noise_spectrum[np.newaxis, :])

        # Compute the inverse STFT
        self.audio = self.istft(de_noised_spectrum * np.exp(1j * np.angle(stft_data)), fft_size, hop_size)

def wiener_filter(self, noise_frames=5, K=1):
        # Set FFT size and hop size
        fft_size = 512
        hop_size = int(fft_size / 2)

        # Compute the Short-Time Fourier Transform (STFT)
        stft_data = self.stft(self.audio, fft_size, hop_size)

        # Estimate the noise spectrum by averaging the magnitudes of the first few frames
        noise_spectrum = np.mean(np.abs(stft_data[:noise_frames]), axis=0)

        # Compute the Wiener filter
        signal_spectrum = np.abs(stft_data)
        wiener_filter = signal_spectrum ** 2 / (signal_spectrum ** 2 + K * noise_spectrum ** 2)

        # Apply the Wiener filter
        de_noised_spectrum = wiener_filter * stft_data

        # Compute the inverse STFT
        self.audio = self.istft(de_noised_spectrum, fft_size, hop_size)

def main():
    audio_processor = AudioProcessor("data/stlkr_ex.wav")
    audio_info_provider = AudioInfoProvider(audio_processor=audio_processor)

    #audio_info_provider.display_info()
    #audio_info_provider.display_waveform()
    #audio_info_provider.display_fft_spectr()

    # normalize & convert to mono
    audio_processor.normalize()
    audio_processor.to_mono()

    audio_processor.spectral_subtraction(alpha=2, beta=0.15)

    # perform equalizing 
    #bands = [(0, 300), (300, 500), (500, 2000), (2000, 3400), (3400, 4800)] # frequencies
    #gains_db = [-12.0, -6.0, 6.0, 3.0, -6.0] # gain in db
    #audio_processor.equalize(bands, gains_db)

    # save processed file
    audio_processor.write_wav_file("data/de_noised.wav")


if __name__ == "__main__":
    main() 