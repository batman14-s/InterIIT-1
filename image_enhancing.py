import os
import time
from turtle import shape
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

class enhance:
  model=None
  def __init__(self):
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    self.model = hub.load(SAVED_MODEL_PATH)

  def preprocess_image(self,img):
      hr_image = tf.convert_to_tensor(img)
      if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
      hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
      hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
      hr_image = tf.cast(hr_image, tf.float32)
      return tf.expand_dims(hr_image, 0)

  def enhanceit(self,img):
    hr_image = self.preprocess_image(img)
    # shape_hr = hr_image.shape
    # print("hr ",shape_hr)

    start = time.time()
    fake_image = self.model(hr_image)
    fake_image = tf.squeeze(fake_image)
    # print(fake_image.shape)
    fake_image=tf.constant(fake_image).numpy().astype(np.uint8)
    # print(fake_image.shape)
    print("Time Taken: %f" % (time.time() - start))
    return fake_image