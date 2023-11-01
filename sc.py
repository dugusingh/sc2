import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

st.title('Image Convolution and Pooling')

# Load the image
@st.cache
def load_image(image_file):
    img = tf.io.read_file(image_file)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, size=[300, 300])
    return img

uploaded_file = st.file_uploader("bhole.jpg", type="jpg")
if uploaded_file is not None:
    image = load_image(uploaded_file)

    # Display original image
    st.subheader('Original Gray Scale Image')
    st.image(image.numpy(), width=300, caption='Original Image')

    # Define the kernel
    kernel = tf.constant([[-1, -1, -1],
                        [1, 8, 1],
                        [-1, -1, -1]])

    # Reformat
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.cast(kernel, dtype=tf.float32)

    # Convolution layer
    conv_fn = tf.nn.conv2d
    image_filter = conv_fn(input=image, filters=kernel, strides=1, padding='SAME')

    # Plot the convolved image
    st.subheader('Convolution')
    st.image(tf.squeeze(image_filter).numpy(), width=300, caption='Convolved Image')

    # Activation layer
    relu_fn = tf.nn.relu
    image_detect = relu_fn(image_filter)

    # Image detection
    st.subheader('Activation')
    st.image(tf.squeeze(image_detect).numpy(), width=300, caption='Activation')

    # Pooling layer
    pool = tf.nn.pool
    image_condense = pool(input=image_detect, window_shape=(2, 2), pooling_type='MAX', strides=(2, 2), padding='SAME')

    st.subheader('Pooling')
    st.image(tf.squeeze(image_condense).numpy(), width=300, caption='Pooling')
