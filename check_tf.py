import tensorflow as tf

print("TensorFlow module:", tf)
print("TensorFlow attributes:", dir(tf))

try:
    print("TensorFlow __file__:", tf.__file__)
except AttributeError:
    print("tf.__file__ not found.")

try:
    print("TensorFlow version:", tf.__version__)
except AttributeError:
    print("tf.__version__ not found.")
