import tensorflow as tf
import os

# 1) Correct path to your original channels_first model
orig_path = r"C:\Users\dheer\Downloads\Detecting-AI-Generated-Fake-Images-main\Detecting-AI-Generated-Fake-Images-main\model_train\mobilemodel_v2"
orig = tf.keras.models.load_model(orig_path)

# 2) Switch Keras to channels_last
tf.keras.backend.set_image_data_format('channels_last')

# 3) Rebuild the model from its config (now defaults to NHWC)
config = orig.get_config()
new = tf.keras.models.Model.from_config(config)
new.set_weights(orig.get_weights())

# 4) Save the new NHWC model alongside the original
out_path = r"C:\Users\dheer\Downloads\Detecting-AI-Generated-Fake-Images-main\Detecting-AI-Generated-Fake-Images-main\model_train\mobilemodel_v2_nhwc"
new.save(out_path, include_optimizer=False, save_format="tf")

print(f"✅ Converted to NHWC and saved to:\n   {out_path}")
