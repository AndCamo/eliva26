import matplotlib.pyplot as plt
import numpy as np
import cv2
cv2.useOptimized() 
import urllib.request



def Malanobis_distance(im, mean, cov):
    data=im.reshape(-1,3)
    data = data - mean
    inv_covmat = np.linalg.inv(cov)
    mhl = np.zeros((data.shape[0],1))
    for i in range(data.shape[0]):
        mhl[i]=data[i,:]@inv_covmat@data[i,:].T
    
    mhl = mhl.reshape(im.shape[0],im.shape[1])
    return mhl


# read samle image from url

imBGR = cv2.imread('Challenge-2/data/train/train_5.jpg')
imRGB = cv2.cvtColor(imBGR, cv2.COLOR_BGR2RGB)

#display image and let the user select a list of RGB color points
plt.figure
# display the image imshow without blocking the code
plt.ion()
plt.imshow(imRGB)   
plt.title('Select color points with left mouse')
plt.show()
print('Select color points with left mouse')
num_points = 10
points = plt.ginput(num_points, timeout=0, show_clicks=True)
plt.close()
points = np.array(points)


# get the RGB values of the selected points
points = np.array(points)
points = points.astype(int)
print('Selected points (round):', points)
colors = imRGB[points[:,1],points[:,0]]
print('Selected colors:', colors)

# computer the mean RGB value
mean_color = np.mean(colors, axis=0)
print('Mean color:', mean_color)

#compute color minus mean RGB value  
diff = colors - mean_color
print('Color minus mean:', diff)


#compute the covariance matrix
C=np.cov(diff.T)
print('Covariance matrix:', C)

plt.ioff()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(colors[:,0], colors[:,1], colors[:,2],c=colors/255)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
plt.title('Selected colors')
plt.show()

mhls = Malanobis_distance(imRGB, mean_color, C)
plt.figure()
plt.imshow(mhls, cmap='gray')
plt.colorbar()
plt.show()

seg = mhls<np.mean(mhls)*0.1
plt.figure()
plt.imshow(seg, cmap='gray')
plt.show()

# segment the image
imRGB_seg = imRGB.copy()
imRGB_seg=imRGB_seg*seg[:,:,np.newaxis]
plt.figure()
plt.imshow(imRGB_seg)
plt.show()


