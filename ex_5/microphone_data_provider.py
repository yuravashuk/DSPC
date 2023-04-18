
import pyaudio
import wave
import numpy as np
import time

class MicrophoneDataProvider:
    def __init__(self, chunk=1024, channels=1, rate=44100, input_device=None) -> None:
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.frames = []
        self.input_device = input_device

    def get_available_devices(self):
        audio = pyaudio.PyAudio()
        num_devices = audio.get_device_count()

        print("Available audio devices:")
        for i in range(num_devices):
            info = audio.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")

        audio.terminate()

    def set_input_device(self, device_index):
        audio = pyaudio.PyAudio()
        num_devices = audio.get_device_count()

        if device_index >= 0 and device_index < num_devices:
            self.input_device = device_index
            audio.terminate()
            return True
        else:
            audio.terminate()
            return False

    def start(self):
        self.audio = pyaudio.PyAudio()

        if self.input_device is None:
            self.stream = self.audio.open(format=pyaudio.paInt16, 
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer=self.chunk)
        else:
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer=self.chunk,
                                          input_device_index=self.input_device)

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.audio.terminate()

    def record(self, duration_sec):
        print("Recording started!")

        self.frames = []
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        print("Recording ended!")

    def write_to_wav(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def get_audio_data(self):
        audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)

        if self.channels > 1:
            audio_data = audio_data.reshape(-1, self.channels)

        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        return audio_data

    def get_framerate(self):
        return self.rate / self.chunk

    def get_samplerate(self):
        return self.rate
