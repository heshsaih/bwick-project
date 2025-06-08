import cv2
import numpy as np
import pyaudio
import librosa
from collections import deque
from statistics import mode, StatisticsError

# Audio processing parameters
RATE = 16000                    # Sample rate for audio processing
CHUNK_SEC = 0.05                # Duration of each audio chunk in seconds
CHUNK = int(RATE * CHUNK_SEC)   # Number of samples in each audio chunk
N_BINS = 64                     # Number of bins for the spectrogram
F0_THRESHOLD = 165.0            # Frequency threshold for F0 detection

# Aliasing parameters
WIN_SIZE = 20                   # Number of buffers to average for aliasing reduction
energy_threshold = 0.005        # Energy threshold for detecting silence

disp_height = 400
disp_width = 800

# Buffers for aliasing
f0_buffer = deque(maxlen=WIN_SIZE)
label_buffer = deque(maxlen=WIN_SIZE)

# Buffer parameters for spectogram
prev_bins = np.zeros(N_BINS, dtype=np.float32)
decay_factor = 0.4
alpha = 0.9

spec_height = disp_height - 50

def draw_spectrum(bins, width, height):
    """
    Draws a spectrogram from the given frequency bins.
    Args:
        bins (np.ndarray): Frequency bins to visualize.
        width (int): Width of the output image.
        height (int): Height of the output image.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bar_width = width // len(bins)
    for i, v in enumerate(bins):
        h = int(v * height)
        x = i * bar_width
        cv2.rectangle(
            img,
            (x, height//2 - h//2),
            (x + bar_width - 1, height//2 + h//2),
            (255, 255, 255),
            -1
        )
    return img

daemon = False
pa = pyaudio.PyAudio()
stream = pa.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

window_name = "Gender Detector"
cv2.namedWindow(window_name)

try:
    while True:
        # Read audio data from the stream
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)

        # Compute RMS (Root Mean Square) energy of the audio samples
        rms = np.sqrt(np.mean(samples**2))
        active = rms >= energy_threshold

        # Compute the FFT and get the magnitude spectrum
        fft_result = np.fft.rfft(samples * np.hanning(len(samples)))
        magnitude = np.abs(fft_result)
        # Logarithmic scaling of the magnitude spectrum
        magnitude = np.log1p(magnitude)
        # Reduction of the magnitude spectrum to N_BINS
        mags = np.array_split(magnitude, N_BINS)
        bins = np.array([m.mean() for m in mags])
        bins = bins / (bins.max() + 1e-6)  # Normalize to [0, 1] range

        if not active:
            bins = np.zeros_like(bins)  # Reset bins if not active
            prev_bins[:] = 0 # Reset previous bins
            f0_buffer.clear()  # Clear F0 buffer when inactive
            label_buffer.clear()  # Clear label buffer when inactive
            smoothed_f0 = 0.0
            gender = "-"
        else:
            bins = alpha * bins + (1 - alpha) * prev_bins
            prev_bins[:] = bins
            # Estimation of fundamental frequency with yin alhorithm to detect the gender of the speaker
            f0 = librosa.yin(
                samples,
                fmin=50.0,
                fmax=500.0,
                sr=RATE,
                frame_length=2048,
                hop_length=128
            )
            f0_nonan = f0[~np.isnan(f0)]
            avg_f0 = float(np.mean(f0_nonan)) if f0_nonan.size > 0 else 0.0
            
            f0_buffer.append(avg_f0)
            smoothed_f0 = np.mean(f0_buffer) if f0_buffer else avg_f0

            # Add the label
            current_label = "Male" if smoothed_f0 <= F0_THRESHOLD else "Female"
            label_buffer.append(current_label)

            try:
                gender = mode(label_buffer)
            except StatisticsError:
                gender = current_label  # Fallback to current label if mode cannot be determined
        
        # Draw the spectrogram
        spectogram_img = draw_spectrum(bins, disp_width, spec_height)
        # Create a canvas to display the spectrogram and text
        canvas = np.zeros((disp_height, disp_width, 3), dtype=np.uint8)
        canvas[0:spec_height, 0:disp_width] = spectogram_img

        cv2.putText(
            canvas,
            f"Gender: {gender}",
            (10, disp_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            canvas,
            f"F0: {smoothed_f0:.2f} Hz",
            (disp_width//2 - 100, disp_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            canvas,
            "Press 'q' to quit",
            (disp_width - 180, disp_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(window_name, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    cv2.destroyAllWindows()