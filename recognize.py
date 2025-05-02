# import the necessary packages
import numpy as np
import imutils
import pickle
import cv2
import os

confidence_threshold = 0.5

def recognize(image_path):
	# load our serialized face detector from disk
	#print("[INFO] loading face detector...")
	detector_path = 'face_detection_model'
	protoPath = os.path.sep.join([detector_path, "deploy.prototxt"])
	modelPath = os.path.sep.join([detector_path,
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load our serialized face embedding model from disk
	#print("[INFO] loading face recognizer...")
	embedding_model_path = 'openface_nn4.small2.v1.t7'
	embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

	# load the actual face recognition model along with the label encoder
	recognizer_path = 'output/recognizer.pickle'
	recognizer = pickle.loads(open(recognizer_path, "rb").read())
	le_path = 'output/le.pickle'
	le = pickle.loads(open(le_path, "rb").read())

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image dimensions
	image = cv2.imread(image_path)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		
		# filter out weak detections
		if confidence > confidence_threshold:
			# compute the (x, y)-coordinates of the bounding box for the
			# face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			# extract the face ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
			
			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the associated
			# probability
			color = (0, 0, 0)
			if proba > confidence_threshold:
				color = (0, 255, 0)
			else:
				color = (0, 0, 255)
				name = 'Unknown'
			text = "{}:{:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				color, 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

	return proba

sum = 0
count = 1
for i in range(1, 53):
	image_path = f'images/{i}-11.jpg'
	probability = recognize(image_path)
	print(f'Person{i} probability is {probability:2f}')
	sum += probability*100
	average = sum/count
	count += 1
	print('The average probability so far is ', average, '\n')