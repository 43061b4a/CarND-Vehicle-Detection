from sklearn.externals import joblib
from pipeline import *
import glob
import cv2
from moviepy.editor import VideoFileClip

start = time.time()

model_file = './models/20170525-160958.pkl'
output_dir = './video_output_images/'

# video_file = './project_video.mp4'
# output_video_file = './out_project_video.mp4'

video_file = './test_video.mp4'
output_video_file = './out_test_video.mp4'

data = joblib.load(model_file)
clf = data['model']
config = data['config']

print("\nsvc : {}".format(type(clf)))
print("configuration : \n\t{}\n".format(config))


def create_image_pipeline(classifier, config):
    def video_image_propcess(img_file):
        return pipeline(img_file, output_dir, True, classifier, config, filepath=False)

    return video_image_propcess


image_pipeline = create_image_pipeline(clf, config)

# load input_video
clip1 = VideoFileClip(video_file)

# apply the pipeline to the video
output_clip = clip1.fl_image(image_pipeline)

# save the new video
output_clip.write_videofile(output_video_file, audio=False)

print("\nProcessing Duration: {} seconds\n\n".format(time.time() - start))
