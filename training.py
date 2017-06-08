import time
import glob
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from feature_extractor import *
from settings import Settings

timestr = time.strftime("%Y%m%d-%H%M%S")


def trainer(data_f):
    # Global Parameters.
    color_space = Settings.color_space  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = Settings.orient  # HOG orientations
    pix_per_cell = Settings.pix_per_cell  # HOG pixels per cell
    cell_per_block = Settings.cell_per_block  # HOG cells per block
    hog_channel = Settings.hog_channel  # Can be 0, 1, 2, or "ALL"
    spatial_size = Settings.spatial_size  # Spatial binning dimensions
    hist_bins = Settings.hist_bins  # Number of histogram bins
    spatial_feat = Settings.spatial_feat  # Spatial features on or off
    hist_feat = Settings.hist_feat  # Histogram features on or off
    hog_feat = Settings.hog_feat  # HOG features on or off

    t = time.time()
    cars = glob.iglob(data_f + '/vehicles/**/*.png')
    car_features = extract_features(cars, cspace=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, extract_spatial_features=spatial_feat,
                                    extract_hist_features=hist_feat, extract_hog_features=hog_feat)
    notcars = glob.iglob(data_f + '/non-vehicles/**/*.png')
    notcar_features = extract_features(notcars, cspace=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, extract_spatial_features=spatial_feat,
                                       extract_hist_features=hist_feat, extract_hog_features=hog_feat)
    t2 = time.time()
    print(round(t2 - t, 2), ' Extract Features Duration')

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # 20% for testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(100))

    clf = Pipeline([('scaling', StandardScaler()),
                    ('classification', LinearSVC(loss='hinge')), ])

    # Check the training time for the SVC
    t = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of classifier = ', round(clf.score(X_test, y_test), 4))

    config = dict(color_space=color_space,
                  spatial_size=spatial_size, hist_bins=hist_bins,
                  orient=orient, pix_per_cell=pix_per_cell,
                  cell_per_block=cell_per_block,
                  hog_channel=hog_channel, spatial_feat=spatial_feat,
                  hist_feat=hist_feat, hog_feat=hog_feat)
    joblib.dump({'model': clf, 'config': config}, 'models/' + timestr + '.pkl')
    print("Classifier, Scaler and setting are saved.")


if __name__ == "__main__":
    data_folder = './data/'
    trainer(data_folder)
