from sklearn.externals import joblib
from pipeline import *
import glob

start = time.time()

model_file = './models/20170525-185124.pkl' # specify the model to use for processing here.
output_dir = './output_images/'

data = joblib.load(model_file)
clf = data['model']
config = data['config']

print("\nsvc : {}".format(type(clf)))
print("configuration : \n\t{}\n".format(config))

input_files = glob.iglob('./test_images/*.jpg')
for file in input_files:
    bounding_boxes_image = pipeline(file, output_dir, True, clf, config, filepath=True)
    image_name = os.path.split(file)[-1]
    cv2.imwrite(os.path.join(output_dir, 'final_' + image_name), bounding_boxes_image)

print("\nProcessing Duration:  {} seconds\n\n".format(time.time() - start))
