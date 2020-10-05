import tensorflow as tf
import config

class Loss:
    def __init__(self):
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.mse = tf.keras.losses.MeanSquaredError()
        model = tf.keras.applications.VGG16(include_top = False)
        self.vgg16 = tf.keras.Model(model.get_layer("block4_conv3").output, trainable = False)
    
    @tf.function
    def reconstruction_loss(self, y_true, y_pred):
        return self.mae(y_true, y_pred)
    
    @tf.function
    def perceptual_loss(self, y_true, y_pred):
        y_true = extract_feature(self.vgg16, y_true)
        y_pred = extract_feature(self.vgg16, y_pred)
        return self.mse(y_true, y_pred) #shouldn't this be MAE?
    
    @tf.function
    def extract_feature(self, vgg16_model, predictions):
        features = predictions
        for layer in vgg16_model.layers:
            features = layer(features)
        return features
    
    @tf.function
    def wrapping_loss(self, frame_0, frame_t, frame_1, backwarp_frames):
        return (self.mae(frame_0, backwarp_frames[0]) +
               self.mae(frame_1, backwarp_frames[1]) +
               self.mae(frame_t, backwarp_frames[2]) +
               self.mae(frame_t, backwarp_frames[3]))
    
    @tf.function
    def smoothness_loss(self, f_01, f_10):
        '''
        f_01 denotes optical flow from frame 0 -> 1, whereas
        f_10 denotes optical flow from frame 1 -> 0
        '''
        delta_f_01 = self.compute_delta(f_01)
        delta_f_10 = self.compute_delta(f_10)
        return delta_f_01 + delta_f_10
    
    @tf.function
    def compute_delta(self, frame):
        x = tf.reduce_mean(tf.abs(frame[:, 1:, :, :] - frame[:, :-1, :, :]))
        y = tf.reduce_mean(tf.abs(frame[:, :, 1:, :] - frame[:, :, :-1, :]))
        return x + y
    
    @tf.function
    def total_loss(self, predictions, loss_values, inputs, frames_t):
        frame_0, frame_1, _ = inputs
        f_01, f_10 = loss_values[:2]
        backwarp_frames = loss_values[2:]
        
        reconstruction = self.reconstruction_loss(frames_t, predictions)
        perceptual = self.perceptual_loss(frames_t, predictions)
        smooth = self.smoothness_loss(f_01, f_10)
        wrap_loss = self.wrapping_loss(frame_0, frame_t, frame_1, backwarp_frames)
        
        delta_loss = (
                    config.Reconstruction * (0.8 * reconstruction)
                    + config.Perception * (0.005*perceptual)
                    + config.Wrap * (0.4*wrap_loss)
                    + config.Smooth * smooth
                    ) 
        return delta_loss, 0.8 * reconstruction, 0.005 * perceptual, 0.4 * smooth, wrap_loss