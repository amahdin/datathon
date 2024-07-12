# DinoDetector: Unearth the Past
![Dinosaurs](https://c02.purpledshub.com/uploads/sites/62/2022/04/GettyImages-724237133-d6e944a.jpg)
## Overview
DinoDetector is an innovative web application that allows users to upload images of dinosaurs and receive instant classification results along with a map showcasing where fossils of the identified species have been found globally. Using advanced deep learning algorithms, DinoDetector aims to educate and engage users in paleontology, making the ancient world of dinosaurs accessible to all.

## Features
- **Image Upload**: Users can easily upload images of dinosaurs to be classified.
- **Dinosaur Classification**: Utilizes a state-of-the-art deep learning model to accurately identify dinosaur species.
- **Fossil Location Mapping**: Displays a global map pinpointing the locations of discovered fossils corresponding to the classified dinosaur.

## Dataset
-  **Dinosaur Image Dataset**: For training our model, we used a dinosaur image dataset available on [Kaggle](https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species) having 15 different species!.

## Resources Used

- **Transfer Learning with TensorFlow**: We used TensorFlow's guide on transfer learning to help train our model efficiently. Check it out [here](https://www.tensorflow.org/tutorials/images/transfer_learning).
- **Machine Learning Model Deployment**: For deploying our machine learning model, this [article on Medium](https://bamblebam.medium.com/how-to-deploy-your-machine-learning-model-using-streamlit-925368b266ad) provided great insights on using Streamlit.
- **Streamlit Documentation**: Streamlit's own [documentation](https://docs.streamlit.io/) was a fantastic resource for setting up our web application.
- **Folium for Mapping**: To create interactive maps showing where dinosaur fossils were found, we used Folium, as detailed in their [documentation](https://python-visualization.github.io/folium/latest/).
- **TensorFlow's ResNet152 Model**: Our deep learning model utilizes TensorFlow’s ResNet152, and you can learn more about it in the [TensorFlow API docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152).
- **Data Augmentation in TensorFlow**: To improve the robustness of our model, we implemented data augmentation techniques outlined in this [TensorFlow tutorial](https://www.tensorflow.org/tutorials/images/data_augmentation).


