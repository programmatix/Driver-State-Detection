import sys
import os

import cv2
import numpy as np
from typing_extensions import List

from Model import ImageAndFilename


import TrainingConstants as tc
from approaches.MediapipeEARMultiFrame.Model import ImageAndFilenameAndContext, GoodBad


def display_frames(images: List[ImageAndFilename], analysed: List[ImageAndFilenameAndContext], current_index):
    frame = images[current_index].image.copy()
    cv2.putText(frame, f'File: {images[current_index].filename}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    a: ImageAndFilenameAndContext = analysed[current_index]
    if a is not None:

        text_list = []
        text_list.extend(a.text_list)


        colour = (255, 255, 255)
        if (a.good == GoodBad.MATCHED_LABEL):
            colour = (0, 255, 0)
        elif (a.good == GoodBad.DID_NOT_MATCH_LABEL):
            colour = (0, 0, 255)
        elif (a.good == GoodBad.LABEL_AMBIGUOUS):
            colour = (255, 0, 0)

        position = 3
        for text in text_list:
            cv2.putText(frame, text, (10, position * 23), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)
            position += 1





        # print(a)
        if a.ai is not None:
            # print(a.ai)
            if a.ai.ai is not None:
                # print(a.ai.ai)
                if a.ai.ai.ai is not None:
                    # print(a.ai.ai.ai)
                    if a.ai.ai.ai.processed is not None:
                        # print(a.ai.ai.ai.processed)
                        # print(a.ai.ai.ai.processed.eye_img_final)
                        if a.ai.ai.ai.processed.eye_img_final is not None:
                            draw = a.ai.ai.ai.processed.eye_img_final
                            draw_y = draw.shape[0]
                            draw_x = draw.shape[1]
                            start_x = frame.shape[1] - draw_x
                            start_y = draw_y
                            frame[start_y:start_y + draw_y, start_x:start_x + draw.shape[1]] = draw

        # if a.ai is not None and a.ai.ai is not None and a.ai.ai.ai is not None and a.ai.ai.ai.processed is not None and a.ai.ai.ai.processed.eye_img_final is not None:
        #     draw = a.ai.ai.ai.processed.eye_img_final
        #     draw_y = draw.shape()[1]
        #     draw_x = draw.shape()[0]
        #     print(draw.shape())
            # frame[0:draw_y, 0:draw_x] = draw

    cv2.imshow('Frames', frame)
