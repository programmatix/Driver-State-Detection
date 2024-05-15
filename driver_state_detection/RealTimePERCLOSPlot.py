import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from time import time

class RealTimePERCLOSPlot:
    def __init__(self):
        self.timestamps = []
        self.ear_scores = []
        self.start_time = time()

    def update_ear_scores(self, ear_score):
        current_time = time() - self.start_time
        self.timestamps.append(current_time)
        self.ear_scores.append(ear_score)
        # Keep only the last 60 seconds of data
        self.timestamps = [t for t in self.timestamps if current_time - t <= 60]
        self.ear_scores = self.ear_scores[-len(self.timestamps):]

    def plot_ear_graph(self):
        fig = plt.figure(figsize=(2, 1))
        # plt.plot(self.timestamps, self.ear_scores, label='EAR Score')
        plt.plot(self.timestamps, self.ear_scores)

        # plt.xlabel('Time (s)')
        # plt.ylabel('EAR')
        # plt.title('Real-time EAR Score Over Last Minute')
        # plt.legend()
        plt.tight_layout()

        canvas = FigureCanvas(plt.gcf())
        canvas.draw()

        buf = canvas.buffer_rgba()
        graph_image = np.asarray(buf, dtype=np.uint8)[:, :, :-1]

        plt.close()

        return graph_image

    def overlay_graph_on_frame(self, frame):
        graph_image = self.plot_ear_graph()
        x_offset = frame.shape[1] - graph_image.shape[1] - 10
        y_offset = 100
        frame[y_offset:y_offset+graph_image.shape[0], x_offset:x_offset+graph_image.shape[1]] = graph_image
