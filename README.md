# IntrusionDetectionSystem

Main Intrusion Detection System implemented in Python for use with OpenCV. 

# Description of Important Files

extract_embeddings.py is the file responsible for taking personnel data and creating serialized pickle file containing 128 dimensional feature vectors of known personnel.

train_model.py is the file responsible for training the SVP overtop these embeddings adding a further classification layer to the IDS. 

recognize.py actually performs the recognition. In its current state, it is simply taking an image input from a subdirectory and performing the face detection and recognition from this input image. 

deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel, and train_model.py are three of the pre-trained and open-sourced models that were used to build the detection and recognition neural networks.
