import cv2
import numpy as np
import math
import os
import sys
import pygame
import copy
from os.path import isdir, isfile, join

def foreground_mask(ins, background):
    old_in = copy.deepcopy(ins)
    out = cv2.absdiff(old_in, background)
    _, out = cv2.threshold(out, 20, 255, cv2.THRESH_BINARY)
    out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return out

def get_shadow(input):
    shadow = copy.deepcopy(input)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow = cv2.cvtColor(shadow, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([1,0,0])
    upper_gray = np.array([250,250,100])
    shadow = cv2.inRange(shadow, lower_gray, upper_gray)
    shadow = cv2.erode(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    # shadow = cv2.dilate(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    return shadow

def get_advanced_shadow(input):
    shadow = copy.deepcopy(input)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow = cv2.erode(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    shadow = cv2.dilate(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    return shadow

def get_hand(input):
    hand = copy.deepcopy(input)
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
    lower_skin_color = np.array([5,100,80])
    upper_skin_color = np.array([17,250,242])
    hand = cv2.inRange(hand, lower_skin_color, upper_skin_color)
    # hand = cv2.erode(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    hand = cv2.dilate(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations = 1)
    return hand

def get_nail(input):
    nail = copy.deepcopy(input)
    nail = cv2.cvtColor(nail, cv2.COLOR_BGR2HSV)
    lower_nail_color = np.array([4,37,54])
    upper_nail_color = np.array([17,110,165])
    nail = cv2.inRange(nail, lower_nail_color, upper_nail_color)
    # hand = cv2.erode(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    hand = cv2.dilate(nail, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations = 1)
    return nail

def count_horizontal_edges(piano, row):
    height, width = piano.shape
    num_edges = 0

    i = 1
    while i < width:
        piano.reshape(height, width)
        pixel = piano[row][i]
        prev_pixel = piano[row][i-1]

        if prev_pixel != pixel and (prev_pixel * pixel) == 0:
            num_edges += 1

        i += 1

    return num_edges

def count_vertical_edges(piano, col, bottom_row, top_row):
    height, width = piano.shape
    num_edges = 0

    heights = []

    i = top_row
    while i < bottom_row:
        piano.reshape(height, width)
        pixel = piano[i][col]
        prev_pixel = piano[i-1][col]

        if prev_pixel != pixel and (prev_pixel * pixel) == 0:
            heights.append(i)
            num_edges += 1

        i += 1

    return num_edges

def find_top_and_bottom_row(piano):
    height, width = piano.shape

    bottom_row = 0
    top_row = 0

    i = height - 6
    while i >= 0:
        num_edges = count_horizontal_edges(piano, i)
        prev_num_edges = count_horizontal_edges(piano, i + 5)
        print("prev_num_horizontal_edges: ", prev_num_edges)

        if num_edges == 16 and prev_num_edges < 16:
            bottom_row = i

        if num_edges < 16 and prev_num_edges == 16:
            top_row = i

        i -= 5

    return (bottom_row, top_row)

def find_left_and_right_row(piano, bottom_row, top_row):
    height, width = piano.shape

    left_row = 0
    right_row = 0

    i = width - 6
    while i >= 0:
        num_edges = count_vertical_edges(piano, i, bottom_row, top_row)
        prev_num_edges = count_vertical_edges(piano, i + 5, bottom_row, top_row)
        print("prev_num_vertical_edges: ", prev_num_edges)

        if num_edges == 2 and prev_num_edges < 2:
            right_row = i

        if num_edges < 2 and prev_num_edges == 2:
            left_row = i

        i -= 5

    return (left_row, right_row)

def play_sound(i):
    if i == 0:
        sound_a.play()
    elif i == 1:
        sound_b.play()
    elif i == 2:
        sound_c.play()
    elif i == 3:
        sound_d.play()
    elif i == 4:
        sound_e.play()
    elif i == 5:
        sound_f.play()
    elif i == 6:
        sound_g.play()

def analyze(piano, hand, shadow, top_row, bottom_row, left_row, right_row):
    height, width = piano.shape

    key_skipped = np.zeros((7), dtype = bool)
    key_state = np.zeros((7), dtype = bool)

    key_size = (right_row - left_row) / 7


    for current_key in range(7):

        row = top_row
        while row < bottom_row:

            col = left_row + (current_key * key_size)
            while col < left_row + ((current_key+1) * key_size):
                col = int(col)
                hand_pixel = hand[row+22][col]
                shadow_pixel = shadow[row][col]

                if shadow_pixel:
                    if 0 <= current_key and current_key < 7:
                        key_skipped[current_key] = True

                if hand_pixel and current_key < 7 and not key_skipped[current_key]:
                    if 0 <= current_key and current_key < 7:
                        key_state[current_key] = True

                col += 1

            row += 1

    # row = top_row  + 5
    # while row < bottom_row:
    #     current_key = -1
    #
    #     col = left_row
    #     while col < right_row:
    #
    #         piano_pixel = piano[row][col]
    #         piano_prev_pixel = piano[row][col-1]
    #         hand_pixel = hand[row+22][col]
    #         shadow_pixel = shadow[row][col]
    #
    #         if not piano_pixel and piano_prev_pixel:
    #             current_key += 1
    #
    #         if shadow_pixel:
    #             if 0 <= current_key and current_key < 7:
    #                 key_skipped[current_key] = True
    #
    #         if hand_pixel and current_key < 7 and not key_skipped[current_key]:
    #             if 0 <= current_key and current_key < 7:
    #                 key_state[current_key] = True
    #
    #         col += 1
    #
    #     row += 1

    for key in key_state:
        if key:
            print(1, end = " ")
        else:
            print(0, end = " ")

    print("")

    counter = 0
    for i in key_state:
        if not global_keystate[counter] and i:
            play_sound(counter)
        global_keystate[counter] = key_state[counter]
        counter += 1

def get_piano(frame):
    piano = copy.deepcopy(frame)

    piano = cv2.cvtColor(piano, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', piano)
    lower_black = np.array([0])
    upper_black = np.array([66])
    mask = cv2.inRange(piano, lower_black, upper_black)
    dilation = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    return dilation

def main():
    cap = cv2.VideoCapture('piano.mp4')

    ret, first_frame = cap.read()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    fgbg = cv2.createBackgroundSubtractorMOG2(50)

    pattern = cv2.imread('pattern.png')
    surf_pattern = cv2.xfeatures2d.SIFT_create(400, edgeThreshold = 9)
    pattern_kp, pattern_desc = surf_pattern.detectAndCompute(pattern, None)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:

            # GET PIANO

            # f_height, f_width, _ = frame.shape
            # ratio = f_width / f_height
            # sz = (int(ratio * 400), 400)
            # frame = cv2.resize(frame, sz)

            surf_frame = cv2.xfeatures2d.SIFT_create(400, edgeThreshold = 9)
            frame_kp, frame_desc = surf_frame.detectAndCompute(frame, None)

            FLANN_INDEX_LINEAR = 0
            FLANN_INDEX_KDTREE = 1
            FLANN_INDEX_KMEANS = 2
            FLANN_INDEX_COMPOSITE = 3
            FLANN_INDEX_KDTREE_SINGLE = 4
            FLANN_INDEX_HIERARCHICAL = 5
            FLANN_INDEX_LSH = 6
            FLANN_INDEX_SAVED = 254
            FLANN_INDEX_AUTOTUNED = 255
            index_params = dict(algorithm=FLANN_INDEX_COMPOSITE, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(frame_desc, pattern_desc, k=2)

            matchesMask = [[0, 0] for i in range(len(matches))]

            good = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
                    good.append(m)

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=2)

            im_matches = cv2.drawMatchesKnn(frame, frame_kp, pattern, pattern_kp, matches, None, **draw_params)
            cv2.imshow('match', im_matches)

            src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([pattern_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, ch = pattern.shape

            rectify = cv2.warpPerspective(frame, M, (w, h))

            cv2.imshow('rectify', rectify)

            piano_template = get_piano(rectify)

            # END

            fgmask = fgbg.apply(rectify)

            shadow = get_advanced_shadow(fgmask)
            hand = get_hand(rectify)
            nail = get_nail(rectify)
            piano = copy.deepcopy(piano_template)

            cv2.imshow('debug', frame)

            frame = cv2.cvtColor(rectify, cv2.COLOR_BGR2GRAY)

            # shadow = cv2.bitwise_and(fgmask, shadow)
            # hand = cv2.bitwise_and(fgmask, hand)
            # hand = cv2.bitwise_not(hand)
            # shadow = cv2.bitwise_and(hand, shadow)
            # hand = cv2.bitwise_not(hand)

            cv2.imshow('shadow', shadow)
            cv2.imshow('hand', hand)
            cv2.imshow('nail', nail)
            cv2.imshow('fg', fgmask)

            hand = cv2.bitwise_or(hand, nail)

            cv2.imshow('hand_and_nail', hand)

            pressedkey = cv2.waitKey(1)
            if pressedkey == 32 or pressedkey == 27:
                while True:
                    pressedkey = cv2.waitKey(1)
                    if pressedkey == 32 or pressedkey == 27:
                        break

            channels = []
            channels.append(hand)
            channels.append(piano)
            # channels.append(piano)
            channels.append(shadow)

            result = cv2.merge(channels)
            cv2.rectangle(result, (305,170), (1015,565), (255,0,0), 3)

            analyze(piano, hand, shadow, 170, 565, 305, 1015)

            cv2.imshow('piano_masks', result)
            cv2.imshow('piano', frame)

            # pressedkey = cv2.waitKey(1)
            # if pressedkey == 32 or pressedkey == 27:
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


global_keystate = np.zeros((7), dtype = bool)

pygame.init()

sound_a = pygame.mixer.Sound('audio/a.wav')
sound_b = pygame.mixer.Sound('audio/b.wav')
sound_c = pygame.mixer.Sound('audio/c.wav')
sound_d = pygame.mixer.Sound('audio/d.wav')
sound_e = pygame.mixer.Sound('audio/e.wav')
sound_f = pygame.mixer.Sound('audio/f.wav')
sound_g = pygame.mixer.Sound('audio/g.wav')

main()
