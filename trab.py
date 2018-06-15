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
    out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)));
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)));
    out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)));
    return out

def get_shadow(input):
    shadow = copy.deepcopy(input)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow = cv2.cvtColor(shadow, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([1,0,0])
    upper_gray = np.array([250,250,100])
    shadow = cv2.inRange(shadow, lower_gray, upper_gray)
    shadow = cv2.erode(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    shadow = cv2.dilate(shadow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    return shadow;

def get_hand(input):
    hand = copy.deepcopy(input)
    hand = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
    lower_skin_color = np.array([1,100,140])
    upper_skin_color = np.array([50,210,250])
    hand = cv2.inRange(hand, lower_skin_color, upper_skin_color)
    hand = cv2.erode(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations = 1)
    hand = cv2.dilate(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)), iterations = 1)
    return hand;

def count_edges(piano, row):
    height, width = piano.shape
    num_edges = 0

    i = 1
    while i < width:
        piano.reshape(height, width)
        pixel = piano[row][i]
        prev_pixel = piano[row][i-1]

        if prev_pixel != pixel and (prev_pixel * pixel) == 0:
            num_edges += 1;

        i += 1

    return num_edges

def find_top_and_bottom_row(piano):
    height, width = piano.shape

    bottom_row = 0
    top_row = 0

    i = height - 6
    while i >= 0:
        num_edges = count_edges(piano, i)
        prev_num_edges = count_edges(piano, i + 5)
        print("prev_num_edges: ", prev_num_edges)

        if num_edges == 16 and prev_num_edges < 16:
            bottom_row = i

        if num_edges < 16 and prev_num_edges == 16:
            top_row = i

        i -= 5

    return (bottom_row, top_row)

def get_piano(frame):
    piano = copy.deepcopy(frame)
    lower_green = np.array([0,0,0])
    upper_green = np.array([10,255,33])
    mask = cv2.inRange(piano, lower_green, upper_green)
    dilation = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    return dilation;

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

def analyze(piano, hand, shadow, top_row, bottom_row):
    height, width = piano.shape

    key_skipped = np.zeros((7), dtype = bool)
    key_state = np.zeros((7), dtype = bool)

    row = bottom_row - 5
    while row > top_row + 5:
        current_key = -1;

        col = 0
        while col < width:

            piano_pixel = piano[row][col]
            piano_prev_pixel = piano[row][col-1]
            hand_pixel = hand[row][col]
            shadow_pixel = shadow[row][col]

            if not piano_pixel and piano_prev_pixel:
                current_key += 1

            if shadow_pixel:
                if 0 <= current_key and current_key < 7:
                    key_skipped[current_key] = True

            if hand_pixel and not key_skipped[current_key]:
                if 0 <= current_key and current_key < 7:
                    key_state[current_key] = True

            col += 1

        row -= 1

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

def main():
    cap = cv2.VideoCapture('piano.mp4')
    cv2.namedWindow('piano')

    ret, first_frame = cap.read()
    piano_template = get_piano(first_frame)

    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('get piano', piano_template)

    bottom_row = 0
    top_row = 0
    bottom_row, top_row = find_top_and_bottom_row(piano_template)

    print("top: ", top_row)
    print("bottom: ", bottom_row)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:

            shadow = get_shadow(frame)
            hand = get_hand(frame)
            piano = copy.deepcopy(piano_template)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('dota', shadow)
            cv2.imshow('cs', hand)

            fg_mask = foreground_mask(frame, first_frame);

            shadow = cv2.bitwise_and(fg_mask, shadow)
            hand = cv2.bitwise_and(fg_mask, hand)

            hand = cv2.bitwise_not(hand)
            shadow = cv2.bitwise_and(hand, shadow)
            hand = cv2.bitwise_not(hand)

            # channels = []
            # channels.append(hand);
            # channels.append(piano);
            # channels.append(shadow);
            #
            # result = cv2.merge(channels);

            analyze(piano, hand, shadow, top_row, bottom_row)

            # cv2.imshow('piano_masks', result)
            cv2.imshow('piano', frame)

            pressedkey = cv2.waitKey(1)
            if pressedkey == 32 or pressedkey == 27:
                break
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
