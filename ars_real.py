import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')


#label_map = {1: 'black', 2:'brown',3:'red',4:'orange',5:'yellow',6:'green',7:'blue',8:'violet',9:'gray',10:'white',11:'gold',12:'silver'}
label_map = {1: 'band'}
train_images_dir = 'dataset_5/train/images'
train_annotations_dir = 'dataset_5/train/annotations'
train_data = object_detector.DataLoader.from_pascal_voc(train_images_dir, train_annotations_dir, label_map=label_map)


val_images_dir = 'dataset_5/validation/images'
val_annotations_dir = 'dataset_5/validation/annotations'
validation_data = object_detector.DataLoader.from_pascal_voc(val_images_dir, val_annotations_dir, label_map=label_map)

test_images_dir = 'dataset_5/test/images'
test_annotations_dir = 'dataset_5/test/annotations'
test_data = object_detector.DataLoader.from_pascal_voc(test_images_dir, test_annotations_dir, label_map=label_map)


spec = object_detector.EfficientDetLite1Spec()


model = object_detector.create(train_data=train_data, model_spec=spec, validation_data=validation_data, epochs=30, batch_size=16, train_whole_model=True)


model.evaluate(test_data)


TFLITE_FILENAME = 'efficientdet-lite-colors_final.tflite'
LABELS_FILENAME = 'band-labels_colors.txt'



from tflite_model_maker.config import ExportFormat
model.export(export_dir='.', tflite_filename=TFLITE_FILENAME, label_filename=LABELS_FILENAME,
             export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

model.evaluate_tflite(TFLITE_FILENAME, test_data)