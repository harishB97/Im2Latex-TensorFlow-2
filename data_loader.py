# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 09:27:10 2022

@author: harish
"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2


class DataGenerator():
    
    def __init__(self, npy_path, image_path, batch_size):
        self.npy_path = npy_path
        self.image_path = image_path
        self.batch_size = batch_size
        self.counter = 0
    
    def __call__(self):
        data_dict = np.load(self.npy_path, allow_pickle=True).tolist()
        for keys in data_dict.keys():
            self.counter = 0
            data_list = data_dict[keys]
            for x, y in data_list:
                if ((len(data_list) - self.counter) % self.batch_size) == 0:
                  break
                x = x[:-4] + '.jpg'
                img = np.asarray(Image.open(os.path.join(self.image_path, x)).convert('YCbCr'))[:,:,0][:,:,None]
                img = img.astype(np.float32)
                seq = np.array(y)
                seq = seq.astype(np.float32)
                
                yield (img, seq[:-1]), seq[1:]
                

def get_dataset(npy_path, image_path, batch_size, filter_predicate=None):

    gen = DataGenerator(npy_path, image_path, batch_size)
    dataset = tf.data.Dataset.from_generator(gen,
                                             output_signature=((tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                                                                  tf.TensorSpec(shape=(None), dtype=tf.float32)),
                                                                tf.TensorSpec(shape=(None), dtype=tf.float32)))
    if filter_predicate is not None:
      dataset = dataset.filter(filter_predicate)
    
    dataset = dataset.padded_batch(batch_size, padding_values=((np.array(255, dtype=np.float32), np.array(0, dtype=np.float32)), np.array(0, dtype=np.float32)),
                                         padded_shapes=((tf.TensorShape([None, None, 1]),
                                                               tf.TensorShape([None])),
                                                               tf.TensorShape([None])))
    return dataset


def show_data(dataset):
    for bn, batch in enumerate(dataset):#.take(1):
        img = batch[0][0]
        for i in range(img.shape[0]):
            cv2.imshow('img', img.numpy()[i].astype(np.uint8))
            key_press = cv2.waitKey(0)
            if key_press & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        if key_press & 0xFF == ord('q'):
            break
