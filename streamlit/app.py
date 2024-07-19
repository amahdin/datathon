import os
import streamlit as st
import pandas as pd
import pylab
import folium
from streamlit_folium import st_folium
import matplotlib
import tensorflow as tf
import numpy as np
import keras
from PIL import Image

# Set page configuration at the very beginning
st.set_page_config(page_title='DinoDetector', page_icon='🦖', layout='wide')

# Print working directory
st.write(f"Current working directory: {os.getcwd()}")

# Load model, set cache to prevent reloading
@st.cache_resource(show_spinner=True)
def load_model():
    model = keras.models.load_model("/models/dinosaur_classifier.keras")
    return model

def load_image(image):
    img = tf.image.decode_image(image, channels=3)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)
    return img

def return_color_generator(NUM_COLORS, cmap='terrain'):
    """
    Generates color map for each dinosaur.
    """
    cm = pylab.get_cmap(cmap)
    for i in range(NUM_COLORS):
        color = cm(1.*i/NUM_COLORS)  # color will now be an RGBA tuple

    # or if you really want a generator:
    cgen = (cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS))

    return cgen

# SOURCE: BugBytes from YouTube
# caching data to avoid reloading data every time the page is refreshed with decorator
@st.cache_data
def read_data(data_loc='data/filtered_dino_fossil_locations.csv'):
    data = list()

    # data frame of dinosaur fossil locations across the world
    dino_fossil_locations = pd.read_csv(data_loc)

    dino_class_names = list(dino_fossil_locations['class_name'].unique())

    # generate colors for each dinosaur class
    dino_class_to_color = dict()
    # returns color generator that we can iterate through
    cgen = return_color_generator(len(dino_class_names))
    for i, rgba in enumerate(cgen):
        dino_class = dino_class_names[i]
        HEXValue = matplotlib.colors.to_hex(rgba)
        dino_class_to_color[dino_class] = HEXValue

    # add colors to the data frame
    dino_fossil_locations['color'] = dino_fossil_locations['class_name'].map(dino_class_to_color)

    return dino_fossil_locations, dino_class_to_color


def main():
    # Create page header and description
    st.header(':t-rex: DinoDetector: Unearth the Past')
    st.markdown('Welcome to our project for the 2024 UCSD DataHacks competition created by Aritra Das, Asif Mahdin, Luke Taylor, and Mandy Xu! Our project classifies an image of a dinosaur using transfer learning from a pre-trained network and finetuned to our dinosaurs dataset. An interactive map is displayed with the highlighted locations of where the classified dinosaur\'s fossils were found in the world.')
    st.markdown('To start, upload an image of a dinosaur or select one from the preselected images. Once classified, explore the interactive map!')

    with st.spinner("Loading Model...."):
        model = load_model()

    classes = [
            "Ankylosaurus",
            "Brachiosaurus",
            "Compsognathus",
            "Corythosaurus",
            "Dilophosaurus",
            "Dimorphodon",
            "Gallimimus",
            "Microceratus",
            "Pachycephalosaurus",
            "Parasaurolophus",
            "Spinosaurus",
            "Stegosaurus",
            "Triceratops",
            "Tyrannosaurus_Rex",
            "Velociraptor",
        ]

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Dropdown for preselected images
    test_imgs_dir = 'test_imgs'
    if os.path.exists(test_imgs_dir):
        preselected_images = os.listdir(test_imgs_dir)
        selected_image = st.selectbox("Or select a preselected image", preselected_images)
    else:
        st.write(f"Directory '{test_imgs_dir}' does not exist.")
        st.write(f"Current working directory: {os.getcwd()}")
        preselected_images = []
        selected_image = None

    pred_class = None

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, use_column_width=True)

        st.write("Predicting Class...")

        with st.spinner("Classifying..."):
            img_tensor = load_image(uploaded_file.read())
            pred = model.predict(img_tensor)
            if pred.size > 0:
                pred_index = np.argmax(pred)
                pred_class = classes[pred_index]
                st.write("Predicted Class:", pred_class)
            else:
                st.write("Prediction array is empty.")

    elif selected_image:
        image_path = os.path.join(test_imgs_dir, selected_image)
        try:
            with open(image_path, 'rb') as image_file:
                image = image_file.read()
            img = Image.open(image_path)
            st.image(img, use_column_width=True)

            st.write("Predicting Class...")

            with st.spinner("Classifying..."):
                img_tensor = load_image(image)
                pred = model.predict(img_tensor)
                if pred.size > 0:
                    pred_index = np.argmax(pred)
                    pred_class = classes[pred_index]
                    st.write("Predicted Class:", pred_class)
                else:
                    st.write("Prediction array is empty.")
        except Exception as e:
            st.write(f"Error loading image: {e}")

    # Dataframe preview of fossil locations
    data, dino_class_to_color = read_data()

    data_class_names = list(data['class_name'].unique())
    m = folium.Map(location=(40.841852219046864, 56.94639806488843), zoom_start=2) # map

    default_label = data_class_names[0]
    if pred_class:
        default_label = pred_class
    else:
        default_label = data_class_names[0]

    # Display map and filters
    st.subheader('Locations of Dinosaur Fossils')
    dinosaur_input = st.multiselect('Select a Dinosaur:', tuple(data_class_names), default=default_label)
    data = data[data['class_name'].isin(dinosaur_input)]
    st.write('Hover over to see general dinosaur names. Click to see the specific dinosaur name.')
    for i, row in data.iterrows():
        location = (row['lat'], row['lng'])
        folium.CircleMarker(location, popup=row['accepted_name'], tooltip=row['class_name'], radius=5, color=row['color'],
        fill_color=row['color']).add_to(m)

    st_folium(m, width=1500)

if __name__ == "__main__":
    main()
