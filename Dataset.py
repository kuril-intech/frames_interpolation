import tensorflow as tf
from pathlib import Path
import os

def decode_img(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def augmentation(image):
    image = tf.image.resize(image, [360, 360])
    image = tf.image.random_crop(image, size = [352, 352, 9])
    image = tf.image.random_flip_left_right(image)
    return image

def load_frames(folder_path, train: bool):
    files = tf.io.matching_files(folder_path + "/*.jpg")
    sampled_indices = tf.random.shuffle(tf.range(12))[:3]
    flip_sequence = tf.random.uniform([], maxval = 1, dtype = tf.int32)
    sampled_indices = tf.where(flip_sequence == 1 and train,
                              tf.sort(sampled_indices, direction = "DESCENDING"),
                              tf.sort(sampled_indices))
    sampled_indices = tf.sort(sampled_indices)
    sampled_files = tf.gather(files, sampled_indices)
    
    frames_0 = decode_img(sampled_files[0])
    frames_1 = decode_img(sampled_files[2])
    frames_t = decode_img(sampled_files[1])
    
    if train:
        frames = augmentation(tf.concat([frames_0, frames_1, frames_t], axis = 2))
        frames_0, frames_1, frames_t = frames[:, :, :3], frames[:, :, 3:6], frames[:, :, 6:9]
    return (frames_0, frames_1, sampled_indices[1]), frames_t



def load_dataset(input_directory, batch_size: int, buffer_size = 1000, train = True, cache = False):
    # autotune = tf.data.experimental.AUTOTUNE
    print(str(os.path.join(input_directory, "*")))
    ds = tf.data.Dataset.list_files(str(os.path.join(input_directory, "*")))
    ds = ds.map(lambda x: load_frames(x, train))#, num_parallel_calls = autotune)
    
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if train:
        ds = ds.shuffle(buffer_size = buffer_size)
    
    ds = ds.batch(batch_size, drop_remainder = True).prefetch(autotune)
    return ds