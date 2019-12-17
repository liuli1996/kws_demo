import tkinter as tk
import threading
from model import SpeechResModel
import torch
from manage_audio import AudioPreprocessor
import collections
import time
import pyaudio
import numpy as np
import librosa
from tqdm import trange


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


class Detector():
    def __init__(self):
        self.audio_processor = AudioPreprocessor()
        self.config = dict(n_labels=12, use_dilation=True, n_layers=13, n_feature_maps=19)
        self.model = SpeechResModel(self.config)
        self.model.load_state_dict(torch.load("../model/res15-narrow-softmax.pt", map_location='cpu'))
        self.labels = ["silence", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    def calculate_probability(self, audio_data):
        x = self.audio_processor.compute_mfccs(audio_data).reshape(1, 101, 40)
        self.model.eval()
        y = self.model(torch.from_numpy(x))
        index = torch.argmax(y)
        label = self.labels[int(index)]
        return y, label

    def evaluate_streaming_audio(self, file_name, output_file, window_size_ms=1000, step_size_ms=200):
        data, sample_rate = librosa.load(file_name, sr=16000)
        sample_step = int(step_size_ms * sample_rate / 1000)
        sample_window = int(window_size_ms * sample_rate / 1000)

        output_labels = []
        self.prob_buffer = collections.deque(maxlen=5)
        for i in trange(0, data.shape[0]-sample_window, sample_step):
            packet = data[i: i + sample_window]
            prob, _ = self.calculate_probability(packet)
            self.prob_buffer.extend(prob.detach().numpy())
            prob_window = np.vstack(self.prob_buffer)
            average_prob = np.mean(prob_window, axis=0)
            max_index = np.argmax(average_prob, axis=-1)
            label = self.labels[int(max_index)]
            prob = average_prob[int(max_index)]
            if prob > 0.7:
                if label == 'silence' and label == '_unknown_':
                    continue
                else:
                    time_ms = i * 1000 / sample_rate
                    output_labels.append({'label': label, 'time': time_ms})
        # 写入txt，不包含silence
        with open(output_file, 'w') as f:
            f.write('label, onset\n')
            for output_label in output_labels:
                f.write('%s, %f\n' % (output_label['label'], output_label['time']))
        return output_labels


class RingBuffer(object):
    """Ring buffer to hold audio from PortAudio"""

    def __init__(self, size=4096):
        self._buf = collections.deque(maxlen=size)

    def extend(self, data):
        """Adds data to the end of buffer"""
        self._buf.extend(data)

    def get(self):
        """Retrieves data from the beginning of buffer and clears it"""
        tmp = bytes(bytearray(self._buf))
        # self._buf.clear()
        return tmp


class HotwordDetector(object):

    def __init__(self):
        self.ring_buffer = RingBuffer(32000)
        self.prob_buffer = collections.deque(maxlen=3)
        self.detector = Detector()
        self.label = "silence"

    def start(self, sleep_time=0.1):

        self._running = True

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(
            input=True,
            output=False,
            channels=1,
            format=pyaudio.paInt16,
            rate=16000,
            frames_per_buffer=1024,
            stream_callback=audio_callback)

        print("detecting...")
        while self._running is True:

            data = self.ring_buffer.get()
            data = buf_to_float(data)
            # print(data)

            if len(data) == 16000:
                prob, _ = self.detector.calculate_probability(data)
                self.prob_buffer.extend(prob.detach().numpy())
                prob_window = np.vstack(self.prob_buffer)
                average_prob = np.mean(prob_window, axis=0)
                max_index = np.argmax(average_prob, axis=-1)
                label = self.detector.labels[int(max_index)]
                probability = average_prob[int(max_index)]

                if probability > 0.7:
                    print(prob)
                    self.label = label
                else:
                    print(prob)
                    self.label = 'silence'

                # prediction, self.label = self.detector.calculate_probability(data)
                # print(prediction, self.label)
            if len(data) == 0:
                time.sleep(sleep_time)
                continue


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()


def result_update():
    var.set(processor.label)
    window.after(100, func=result_update)


if __name__ == "__main__":
    processor = HotwordDetector()
    window = tk.Tk()
    window.title('keyword spotting')
    window.geometry('300x200')
    var = tk.StringVar()
    l = tk.Label(window, textvariable=var, font=('Consolas', 48))
    l.pack(expand='yes')

    thread_it(func=processor.start)
    window.after(100, func=result_update)
    window.mainloop()

