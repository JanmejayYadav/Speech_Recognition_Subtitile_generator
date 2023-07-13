import glob
import cv2
import os
import WordVideo
import dlib
import time
import ROI
import TPE
import Gabor
import Features
a=time.time()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # path of "shape_predictor_68_face_landmarks.dat"
#VideoPath ='/Users/jay/Documents/**/*.mp4'  # video path
VideoPath ='/Users/jay/Documents/DisserationProject/lrs2_v1_partaa/mvlrs_v1/main/**/*.mp4'
Frame = '/Users/jay/Documents/FeatureExtraction/Picture/'# path to store pictures
MouthPath = '/Users/jay/Documents/FeatureExtraction/mouth/'  # path to store mouth region
GaborPath = '/Users/jay/Documents/FeatureExtraction/Gabor/'# path to store Gabor features
SheetPath = '/Users/jay/Documents/FeatureExtraction/Sheet/' # path to store lip features
FeaturesPath = '/Users/jay/Documents/FeatureExtraction/Features/'  # path to store lip features


WordVideo.Frame(detector,predictor,VideoPath,Frame,MouthPath,GaborPath,SheetPath,FeaturesPath)


b=time.time()
print("Time = ",b-a)
