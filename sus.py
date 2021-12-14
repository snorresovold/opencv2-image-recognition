import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance
import math



img = cv2.imread('FHtLk.png',0)


_,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY_INV)

labels, stats = cv2.connectedComponentsWithStats(img, 8)[1:3]



for label in np.unique(labels)[1:]:

    arrow = labels==label

    indices = np.transpose(np.nonzero(arrow)) #y,x

    dist = distance.cdist(indices, indices, 'euclidean')


    far_points_index = np.unravel_index(np.argmax(dist), dist.shape) #y,x


    far_point_1 = indices[far_points_index[0],:] # y,x
    far_point_2 = indices[far_points_index[1],:] # y,x


    ### Slope
    arrow_slope = (far_point_2[0]-far_point_1[0])/(far_point_2[1]-far_point_1[1])  
    arrow_angle = math.degrees(math.atan(arrow_slope))

    ### Length
    arrow_length = distance.cdist(far_point_1.reshape(1,2), far_point_2.reshape(1,2), 'euclidean')[0][0]


    ### Thickness
    x = np.linspace(far_point_1[1], far_point_2[1], 20)
    y = np.linspace(far_point_1[0], far_point_2[0], 20)
    line = np.array([[yy,xx] for yy,xx in zip(y,x)])
    thickness_dist = np.amin(distance.cdist(line, indices, 'euclidean'),axis=0).flatten()

    n, bins, patches = plt.hist(thickness_dist,bins=150)

    thickness = 2*bins[np.argmax(n)]

    print(f"Thickness: {thickness}")
    print(f"Angle: {arrow_angle}")
    print(f"Length: {arrow_length}\n")
    plt.figure()
    plt.imshow(arrow,cmap='gray')
    plt.scatter(far_point_1[1],far_point_1[0],c='r',s=10)
    plt.scatter(far_point_2[1],far_point_2[0],c='r',s=10)
    plt.scatter(line[:,1],line[:,0],c='b',s=10)
    plt.show()