# -*- coding: utf-8 -*-
"""
@author: DarthVictor
"""

import numpy as np
import math
from skimage import io, img_as_float
image_rgb = io.imread('parrots.jpg')

import pylab
pylab.imshow(image_rgb)

image = img_as_float(image_rgb)

w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))


from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



def checkNumColors(n_colors):
    kmeans = KMeans(n_clusters=n_colors, init='k-means++', random_state=241).fit(image_array)
    labels = kmeans.predict(image_array)
    
    
  
    
    mean_values = np.zeros((n_colors, d))
    median_values = np.zeros((n_colors, d))
    
    for cluster_i in range(0, n_colors):
        cluster_dots = image_array[labels == cluster_i]
        r_array = cluster_dots[:,0]
        g_array = cluster_dots[:,1]
        b_array = cluster_dots[:,2]
        mean_values[cluster_i] = np.array([np.mean(r_array), np.mean(g_array), np.mean(b_array)])
        median_values[cluster_i] = np.array([np.median(r_array), np.median(g_array), np.median(b_array)])
    
    image_mean = np.zeros((w, h, d))
    image_median = np.zeros((w, h, d))
    
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image_mean[i][j] = mean_values[labels[label_idx]]
            image_median[i][j] = median_values[labels[label_idx]]
            label_idx += 1
                 
#    plt.figure(1)
#    plt.clf()
#    plt.axis('off')
#    plt.title('Quantized image, mean')
#    plt.imshow(image_mean)
#
#    plt.figure(2)
#    plt.clf()
#    plt.axis('off')
#    plt.title('Quantized image, median')
#    plt.imshow(image_median)
        
    return (image_mean, image_median)


def psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    return 10 * math.log10(float(1) / mse)


for n_colors_i in range(10, 0, -1):
    print('>>>>>>', n_colors_i)
    (image_mean, image_median) = checkNumColors(n_colors_i)    
    print(psnr(image_mean, image), psnr(image_median, image))