import os
import sys
import random

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import tensorflow as tf


def _load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        return image
#         return image.astype(np.float32)
    return None

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=value))

def _build_examples_list(img_dir, csv_dir):
    df = pd.read_csv(csv_dir)
    examples = []
    for i, row in df.iterrows():
        img_name = row['img_name']
        latex_seq = row['latex_seq']
        filepath = os.path.join(img_dir, img_name+".jpg")
        example = {
            'path': filepath,
            'img_name': img_name, 
            'latex_seq': np.array([int(x) for x in latex_seq.strip('[]').split(', ')]) 
        }
        examples.append(example)
            
    return examples

def _split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

# def _get_examples_share(examples, training_split):
#     examples_size = len(examples)
#     len_training_examples = int(examples_size * training_split)

#     return np.split(examples, [len_training_examples])

def _write_tfrecord(examples, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)
    empty_images = []
    failed = False
    for example in tqdm(examples):
        try:
            image = _load_image(example['path'])
            if image is not None:
                encoded_image_string = cv2.imencode('.jpg', image)[1].tobytes()
                feature = {
                    'image': _bytes_feature(tf.compat.as_bytes(encoded_image_string)),
                    'latex_seq_in': _int64_feature(example['latex_seq'][:-1]),
                    'latex_seq_out': _int64_feature(example['latex_seq'][1:])
                }

                tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
            else:
                empty_images.append(example['img_name'])
        except Exception as inst:
            print(inst)
            cmd = input("Continue? (y/n):")
            if cmd == 'y':
                pass
            else:
                failed = True
                break
    print(len(empty_images), empty_images)
    writer.close()
    return failed

def _write_sharded_tfrecord(examples, number_of_shards, output_filename):
    sharded_examples = _split_list(examples, number_of_shards)
    for count, shard in tqdm(enumerate(sharded_examples, start = 1)):
        output_filename = '{0}_{1:02d}of{2:02d}.tfrecord'.format(
            base_output_filename,
            count,
            number_of_shards 
        )
        failed = _write_tfrecord(shard, output_filename)
        if failed:
            break

img_dir = os.path.join('.//altered_images_1//')
csv_dir = os.path.join(r'validate.csv')
base_output_filename = os.path.join(r'./val_100K')
number_of_shards = 1
examples = _build_examples_list(img_dir, csv_dir)
# training_examples, test_examples = _get_examples_share(examples, app['TRAINING_EXAMPLES_SPLIT']) # pylint: disable=unbalanced-tuple-unpacking

print("Creating training shards", flush = True)
_write_sharded_tfrecord(examples, number_of_shards, base_output_filename)
print("\n", flush = True)