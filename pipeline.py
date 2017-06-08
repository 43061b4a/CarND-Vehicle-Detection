import os
from utils import *


def pipeline(file, output_dir, save_files, svc, config, filepath=False):
    if filepath:
        print("Processing: " + file)
        img = cv2.imread(file)
        image_name = os.path.split(file)[-1]
    else:
        img = file
        image_name = 'video_'

    scales = [(64, 64), (96, 96), (128, 128), (192, 192)]
    overlaps = [0.75, 0.75, 0.75, 0.75]
    y_start_stops = [[400, 500], [400, 500], [400, 600], [400, 600]]
    x_start_stops = [[None, None], [None, None], [None, None], [None, None]]
    colors = [(255, 0, 0), (128, 128, 0), (0, 128, 128), (0, 0, 0)]

    color_space = config['color_space']  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = config['orient']  # HOG orientations
    pix_per_cell = config['pix_per_cell']  # HOG pixels per cell
    cell_per_block = config['cell_per_block']  # HOG cells per block
    hog_channel = config['hog_channel']  # Can be 0, 1, 2, or "ALL"
    spatial_size = config['spatial_size']  # Spatial binning dimensions
    hist_bins = config['hist_bins']  # Number of histogram bins
    spatial_feat = config['spatial_feat']  # Spatial features on or off
    hist_feat = config['hist_feat']  # Histogram features on or off
    hog_feat = config['hog_feat']  # HOG features on or off

    windowed_image = np.copy(img)
    hot_windowed_image = np.copy(img)
    heat = np.zeros_like(img[:, :, 0]).astype(np.uint8)

    for scale, overlap, color, y_start_stop, x_start_stop in zip(scales, overlaps, colors, y_start_stops,
                                                                 x_start_stops):

        windows = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                               xy_window=scale, xy_overlap=(overlap, overlap))

        hot_windows = search_windows(img, windows, svc, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        heat = add_heat(heat, hot_windows)

        if save_files:
            # Update windowed images
            hot_windowed_image = draw_boxes(hot_windowed_image, hot_windows, color=color, thick=6)
            windowed_image = draw_boxes(windowed_image, windows, color=color, thick=6)

    if save_files:
        f = plt.figure(figsize=(8, 6))
        plt.subplot(121)
        plt.imshow(hot_windowed_image)
        plt.title('Hot Windowed Image')
        plt.subplot(122)
        plt.imshow(windowed_image)
        plt.title('Windowed Image')
        f.tight_layout()
        f.savefig(os.path.join(output_dir, 'windows_' + time.strftime("%Y%m%d-%H%M%S") + image_name),
                  bbox_inches='tight', pad_inches=0)
        f.clear()

    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if save_files:
        f = plt.figure(figsize=(8, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        f.tight_layout()
        f.savefig(os.path.join(output_dir, 'heatmap_' + time.strftime("%Y%m%d-%H%M%S") + image_name),
                  bbox_inches='tight', pad_inches=0)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img
