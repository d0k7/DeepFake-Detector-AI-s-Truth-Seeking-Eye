import tensorflow as tf
import tf2onnx

# 1) Load your original NCHW‐based model
orig = tf.keras.models.load_model(
    r"C:\Users\dheer\Downloads\Detecting-AI-Generated-Fake-Images-main\Detecting-AI-Generated-Fake-Images-main\model_train\mobilemodel_v2"
)

# 2) Tell tf2onnx that the model input is (None, 3, 224, 224)
spec = (tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input"),)

# 3) Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(
    orig,
    input_signature=spec,
    opset=14,
    output_path="mobilemodel_v2.onnx"
)

print("✅ ONNX model saved to mobilemodel_v2.onnx")
