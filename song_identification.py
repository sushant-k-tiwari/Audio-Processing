import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

def load_audio(file_path, start_time, duration):
    audio = AudioSegment.from_file(file_path)
    start_ms = start_time * 1000
    end_ms = start_ms + duration * 1000
    audio_segment = audio[start_ms:end_ms]
    samples = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate
    return samples, sample_rate

def compute_spectrum(samples, sample_rate):
    fft_result = np.fft.fft(samples)
    frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)
    magnitude_spectrum = np.abs(fft_result)

    return frequencies, magnitude_spectrum

def plot_spectrum(frequencies, magnitude_spectrum, label=None):
    plt.plot(frequencies, magnitude_spectrum, label=label)

def find_best_match(file1_samples, file2_samples, sample_rate):
    cross_corr = np.correlate(file1_samples, file2_samples, mode='full')
    lags = np.arange(-len(file1_samples) + 1, len(file1_samples))
    time_seconds = lags / sample_rate
    threshold = 0.95 * np.max(cross_corr)
    high_corr_indices = np.where(cross_corr > threshold)[0]
    if len(high_corr_indices) == 0:
        return -1, -1
    best_match_time = time_seconds[high_corr_indices[0]]
    best_match_score = cross_corr[high_corr_indices[0]]
    
    print("\nCross-Correlation")
    for lag, value in zip(time_seconds, cross_corr):
        if lag % 1 == 0:
            print(f"Lag: {lag} seconds, Cross-Correlation Value: {value}")

    return best_match_time, best_match_score

def main():
    input_audio = 'pp.wav'  
    duration = 3 #Segment Size
    best_match_time = -1
    best_match_score = -1
    best_match_segment = -1

    for segment_num, start_time in enumerate(range(0, len(AudioSegment.from_file(input_audio)) // 1000, duration)):
        samples1, sample_rate1 = load_audio(input_audio, start_time, duration)
        frequencies1, magnitude_spectrum1 = compute_spectrum(samples1, sample_rate1)
        plot_spectrum(frequencies1, magnitude_spectrum1, label=f'Test Sound - Segment {segment_num+1}')

        test_audio = 'Recording.wav'  
        samples2, sample_rate2 = load_audio(test_audio, 0, 5)
        frequencies2, magnitude_spectrum2 = compute_spectrum(samples2, sample_rate2)
        plot_spectrum(frequencies2, magnitude_spectrum2, label='Input Audio')

        match_time, match_score = find_best_match(samples1, samples2, sample_rate1)

        if match_score > best_match_score:
            best_match_score = match_score
            best_match_time = match_time
            best_match_segment = segment_num

        plt.title(f'Frequency Spectrum Comparison - Segment {segment_num+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.show()

    if best_match_time != -1:
        print(f'\nBest Match Segment: {best_match_segment+1} with Cross Correlation Match Score: {best_match_score}')
    else:
        print('No match found.')

if __name__ == "__main__":
    main()
