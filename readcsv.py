import pandas as pd
import cv2
from shapely.geometry import Polygon

# put csv data in dataframe
df = pd.read_csv('WashingtonOBRace/WashingtonOBRace/corners.csv', names=["img", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
# drop second and third gates, only keep closest gate
df.drop_duplicates(subset='img', keep='first', inplace=True)
df.reset_index(inplace=True, drop=True)

'''
for i in range(0,len(df)):
    df['img'][i] = df['img'][i][4:-4]
    
# sort by name
df["img"] = df["img"].astype(str).astype(int)
df.sort_values('img', inplace=True)
'''

# find img_*.png in files[i][34:]

# calculate IoU
def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

for i in range(0,len(df)):
    corners_ground = np.float32([ [df['x1'][i],df['y1'][i]],[df['x2'][i],df['y2'][i]],[df['x3'][i],df['y3'][i]],[df['x4'][i],df['y4'][i]] ]).reshape(-1,1,2)
    img1 = cv2.polylines(img1, [np.int32(corners_ground)], True, (0,0,255), 1, cv2.LINE_AA)


    #box_1 = [[511, 41], [577, 41], [577, 76], [511, 76]]
    #box_2 = [[544, 59], [610, 59], [610, 94], [544, 94]]
    box_1 = [dst[0].tolist()[0], dst[1].tolist()[0], dst[2].tolist()[0], dst[3].tolist()[0]]
    box_2 = [corners_ground[0].tolist()[0], corners_ground[1].tolist()[0], corners_ground[2].tolist()[0], corners[3].tolist()[0]]

    print(calculate_iou(box_1, box_2))