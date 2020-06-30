import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

## Precalculations

# collect image files and make image list
img_dir = "WashingtonOBRace/WashingtonOBRace"
data_path = os.path.join(img_dir,'img_*.png')
files = glob.glob(data_path) # list image paths
data = []                    # image list
for file in files:
    img = cv2.imread(file)
    data.append(img)


# Template image and mask
img1 = cv2.imread("WashingtonOBRace/WashingtonOBRace/img_431.png", cv2.IMREAD_GRAYSCALE)
img1_mask = cv2.imread("WashingtonOBRace/WashingtonOBRace/mask_431.png", cv2.IMREAD_GRAYSCALE)
# draw template gate
corners = np.float32([ [116,136],[281,138],[278,300],[95,306] ]).reshape(-1,1,2) #img_431
img1 = cv2.polylines(img1, [np.int32(corners)], True, (0,0,255), 1, cv2.LINE_AA)
# ORB keypoints and descriptors of template image
orb = cv2.ORB_create(nfeatures=1000)
kp1, desc1 = orb.detectAndCompute(img1, img1_mask)
img1 = cv2.drawKeypoints(img1, kp1, None)
# Brute-Force Feature Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

## Loop through image list and apply gate detection test algorithm

for i in range(0,len(files)-1):
    start_time = time.time()

    # read image and determine ORB keypoints+descriptors
    img2 = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    img2 = cv2.drawKeypoints(img2, kp2, None)

    # Brute-Force Feature Matching
    matches = bf.match(desc1,desc2)
    matches = sorted(matches, key=lambda x:x.distance)
    print(len(matches))

    # Extract the matched keypoints
    src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Find homography matrix and make a perspective transform
    hmatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    dst = cv2.perspectiveTransform(corners,hmatrix)
    end_time = time.time()
    #print("Found a gate in {0} seconds.".format(end_time - start_time))

    # draw found gate
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)


    #print("Matches found - %d" % (len(matches)))
    # draw first 10 match lines
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    cv2.imshow("gate detection", result)
    cv2.waitKey(500)
    cv2.destroyAllWindows()