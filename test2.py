import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

## Precalculations

# collect image files and make image list
img_dir = "WashingtonOBRace/WashingtonOBRace"
data_path = os.path.join(img_dir,'img_*.png')
files = sorted(glob.glob(data_path)) # sorted list image paths
data = []                    # image list
for file in files:
    img = cv2.imread(file)
    data.append(img)

# put csv data in dataframe
df = pd.read_csv('WashingtonOBRace/WashingtonOBRace/corners.csv', names=["img", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
# drop second and third gates, only keep closest (detected) gate
df.drop_duplicates(subset='img', keep='first', inplace=True)
df.reset_index(inplace=True, drop=True)

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


# FLANN Feature Matcher
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 12, # 12 6
                   key_size = 20,     # 20 12
                   multi_probe_level = 2) #2 1
search_params = dict(checks = 100)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Feature match count condition
MIN_MATCH_COUNT = 12

# calculate IoU function
def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

computation_time = []
iou_list = []
## Loop through image list and apply gate detection test algorithm

for i in range(0,len(files)):
    # start computation time
    start_time = time.time()
    corners_ground = np.float32([ [df['x1'][i],df['y1'][i]],[df['x2'][i],df['y2'][i]],[df['x3'][i],df['y3'][i]],[df['x4'][i],df['y4'][i]] ]).reshape(-1,1,2)

    # read image and determine ORB keypoints+descriptors
    img2 = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    img2 = cv2.drawKeypoints(img2, kp2, None)

    # Brute-Force Feature Matching
    matches = flann.knnMatch(desc1,desc2,k=2)

    # ratio test as per Lowe's paper
    good = []
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        except ValueError:
            pass


    if len(good)>MIN_MATCH_COUNT:

        # Extract the matched keypoints
        src_pts  = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Find homography matrix and make a perspective transform
        hmatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        dst = cv2.perspectiveTransform(corners,hmatrix)
        # end computation time
        end_time = time.time()
        
        # draw ground truth gate (blue)
        img2 = cv2.polylines(img2, [np.int32(corners_ground)], True, (255,0,0), 1, cv2.LINE_AA)
        # draw found gate (red)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
        # calculate IoU
        box_1 = [dst[0].tolist()[0], dst[1].tolist()[0], dst[2].tolist()[0], dst[3].tolist()[0]]
        box_2 = [corners_ground[0].tolist()[0], corners_ground[1].tolist()[0], corners_ground[2].tolist()[0], corners[3].tolist()[0]]
        try:
            iou = calculate_iou(box_1, box_2)
        except:
            iou = 0
        print("%d matches are found. IoU: %s computation time: %s " % (len(good), iou, (end_time - start_time)))
        computation_time.append((end_time - start_time))
        iou_list.append(iou)
    else:
        print("Not enough matches found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = None,
                    matchesMask = matchesMask,
                    flags = 2)

    #print("Matches found - %d" % (len(matches)))
    # draw first 10 match lines
    result = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    cv2.imshow("gate detection", result)
    cv2.waitKey(1) # change window open time in ms
    cv2.destroyAllWindows()


print("average computation time = %s seconds" % (sum(computation_time)/len(computation_time)))
print("min computation time = %s seconds" % (min(computation_time)))
print("max computation time = %s seconds" % (max(computation_time)))
plt.plot(iou_list)
plt.ylabel('IoU')
plt.xlabel('image number')
plt.show()