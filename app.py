from FlaskObjectDetection.utils import visualization_utils as vis_util
from FlaskObjectDetection.utils import label_map_util
from FlaskObjectDetection import text_reco
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn
import torch.backends.cudnn as cudnn
import skimage
import argparse
from flask_ngrok import run_with_ngrok
from skimage import io
import json
import zipfile
from collections import OrderedDict
import cv2

import FlaskObjectDetection.text_reco.models.craft.craft_utils as craft_utils
import FlaskObjectDetection.text_reco.models.craft.imgproc as imgproc
import FlaskObjectDetection
from FlaskObjectDetection.text_reco.models.craft.craft import CRAFT
from FlaskObjectDetection.text_reco.models.craft.craft_reader import CraftReader
from FlaskObjectDetection.text_reco.boxdetect.box_detection import BoxDetect
from FlaskObjectDetection.text_reco.models.crnn.crnn_run import CRNNReader

sys.path.append("..")


MODEL_NAME = 'ssd'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'label_map.pbtxt')
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

run_with_ngrok(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DETECT_FOLDER'] = 'uploads/detect/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET','POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))


@app.route('/uploads/<filename>', methods=['GET','POST'])
def uploaded_file(filename):
    detection_box_location = []
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(
        PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image = image.rotate(270)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                detection_box_location = vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                if(len(detection_box_location) != 0):
                    im = im.crop(
                        (detection_box_location[0], detection_box_location[2], detection_box_location[1], detection_box_location[3]))
                    im.save('uploads/detect/' + filename)
                    send_from_directory(app.config['DETECT_FOLDER'], filename)
                    crr = CraftReader('uploads/detect/'+filename)
                    crnn = CRNNReader() 
                    boxes, img_res = crr.boxes_detect()
                    results = {}
                    for _, tmp_box in enumerate(boxes):
                        tmp_img = img_res
                        tmp_img = Image.fromarray(tmp_img.astype('uint8')).convert('L')
                        tmp_img = crnn.transformer(tmp_img)
                        tmp_img = tmp_img.view(1, *tmp_img.size())
                        tmp_img = Variable(tmp_img)
                        results['{}'.format(_)] = crnn.get_predictions(tmp_img)
                        print(results)
                    
                    return results
                    
                    
                    #return redirect('/')
                else:
                    im.save('uploads/' + filename)
                    send_from_directory(app.config['UPLOAD_FOLDER'], filename)
                    noresult = OrderedDict()
                    noresult["0"] = "NO"
                    return noresult


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

