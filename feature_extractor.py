import cv2
import time
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
import glob

hog_extractor_global = None

timestr = time.strftime("%Y%m%d-%H%M%S")


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if visualize is True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     extract_spatial_features=True, extract_hist_features=True, extract_hog_features=True):
    # Create a list to append feature vectors to
    features = []

    # hog_extract_manager = None

    # Iterate through the list of images
    for file in imgs:

        file_features = []

        # Read in each one by one
        img = mpimg.imread(file)

        img = (img * 255).astype(np.uint8)

        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(img)

        if extract_spatial_features:
            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if extract_hist_features:
            # Apply color_hist() also with a color space option now
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            file_features.append(hist_features)

        if extract_hog_features:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)

            file_features.append(hog_features)

        # Append the new feature vector to the features list
        features.append(np.concatenate(file_features))

    # Return list of feature vectors
    return features


if __name__ == "__main__":
    cars = glob.glob('./data/vehicles/**/*.png')
    notcars = glob.glob('./data/non-vehicles/**/*.png')
    car = mpimg.imread(cars[10])
    notcar = mpimg.imread(notcars[10])

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Car')
    plt.imshow(car)

    plt.subplot(1, 2, 2)
    plt.imshow(notcar)
    plt.title('Not Car')

    fig.savefig('./output_images/sample_image.png')
    fig.clear()

    img = (car * 255).astype(np.uint8)

    from skimage.feature import hog

    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    hog_feat, hog_image = hog(img1[:, :, 0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                              visualise=True)
    # Preview
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HoG - YCrCb')
    fig.savefig('./output_images/hog_image1.png', bbox_inches='tight', pad_inches=0)
    fig.clear()

    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    hog_feat, hog_image = hog(img2[:, :, 0], orientations=8, pixels_per_cell=(10, 10), cells_per_block=(4, 4),
                              visualise=True)
    # Preview
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('RGB')
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HoG - YCrCb')
    fig.savefig('./output_images/hog_image2.png', bbox_inches='tight', pad_inches=0)
    fig.clear()
