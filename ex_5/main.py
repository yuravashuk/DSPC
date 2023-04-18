
from audio_processor import AudioProcessor
from audio_info_provider import AudioInfoProvider
from microphone_data_provider import MicrophoneDataProvider

def main():
    # init basic classes
    audio_processor = AudioProcessor("ex_5/data/output_original_M.wav")
    audio_info_provider = AudioInfoProvider(audio_processor=audio_processor)
    mic_data_provider = MicrophoneDataProvider()

    # setup default microphone device
    #mic_data_provider.get_available_devices()
    #mic_data_provider.set_input_device(16)

    # record 5 sec
    mic_data_provider.start()
    mic_data_provider.record(5)

    # save original record
    mic_data_provider.write_to_wav("ex_5/data/output_original_M.wav")

    # submit data from microphone to audio processor
    audio_processor.submit_buffer(mic_data_provider=mic_data_provider)

    # apply equalizer
    bands = [(0, 300), (300, 500), (500, 2000), (2000, 3400), (3400, 4800)] # frequencies
    gains_db = [2.0, 8.0, -12.0, -6.0, -12.0] # gain in db
    audio_processor.equalize(bands, gains_db)

    # save processed record
    audio_processor.write_wav_file("ex_5/data/output_processed_M.wav")

    # disable mic wrapper 
    mic_data_provider.stop()

if __name__ == "__main__":
    main() 