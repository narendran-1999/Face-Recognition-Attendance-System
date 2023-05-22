from flask import Flask, render_template, Response, send_from_directory, jsonify, request
import cv2
import numpy as np
from imutils.video import VideoStream, FileVideoStream
import csv
import tensorflow as tf
from keras_facenet import FaceNet
from keras.models import load_model
import datetime


app = Flask(__name__, static_folder='static')
vs = None
file_source = "test_vids/camsim.mp4"
src = 0

facenet_model = FaceNet()
fmodel = load_model("8class_model_50epoch.h5")

#fmodel = tf.keras.models.clone_model(fmodel)
#fmodel.set_weights(fmodel.get_weights())
#fmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#fmodel = tf.keras.utils.multi_gpu_model(fmodel)

prototxt = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)


label = []
with open('students.csv', 'r') as f:
    reader = csv.reader(f)
    
    for row in reader:
        data = [row[0],0]
        label.append(data)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

@app.route("/collect")
def collect():
    return render_template("collect.html")

@app.route("/archive")
def archive():
    return render_template("archive.html")

def get_frame():
    
    '''src = int(input("Source:\n0 - File\n1 - Webcam\n"))'''

    global file_source,src

    # Start the video stream
    vs = VideoStream(src=0).start() if src==1 else FileVideoStream(file_source).start()

    # Get the video writer ready
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fps = vs.stream.get(cv2.CAP_PROP_FPS)
    #width = int(vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #out = cv2.VideoWriter("outvid.mp4", fourcc, fps, (width, height))


    while True:
        frame = vs.read()
        if frame is None:
            break

        markFaces(frame)
        
        # Convert the captured frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)


        #out.write(frame)#  write to video


        # Return the frame as a bytes-like object
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def markFaces(frame):
        
        global net

        (h, w) = frame.shape[:2]
        #blob = cv2.dnn.blobFromImage(frame, 1.0, (h, h), (104.0, 177.0, 123.0))
        #blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
        blob = cv2.dnn.blobFromImage(frame, 0.5, (h,h), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence < 0.6:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            img = frame[startY:endY,startX:endX]

            face_id = recFace(img)
            if face_id != -1:
                face = label[face_id][0]
            else:
                face = "Unknown"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame,face,(startX-10,startY-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)

            with open("att_facenet.csv", "a", newline="\n") as csvfile:
                
                time = str(datetime.datetime.now())
                
                row = [time,face]

                if face_id == -1 or label[face_id][1] == 0:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(row)

                        if face_id != -1:
                            label[face_id][1] = 1

def recFace(image):

    #lbl = ""
    max = -1

    if np.any(image):

        #try:

            global facenet_model, fmodel

            image = cv2.resize(image,(224,224))
            image = tf.keras.preprocessing.image.img_to_array(image)

            #image = tf.constant(image)
            #image = tf.keras.backend.cast(image, 'float32')
            #image = tf.keras.backend.reshape(image, (1, 224, 224, 3))
            #image = tf.keras.backend.concatenate([image, image], axis=0)

            imgs = [image]

            embeddings = facenet_model.embeddings(imgs)

            embeddings = tf.constant(embeddings)
            embeddings = tf.keras.backend.cast(embeddings, 'float32')

            pred = fmodel.predict(embeddings)
            max = np.argmax(pred[0])

            print(pred)

            if pred[0][max] < 0.6:
                #lbl = "Unknown"
                max = -1

        #except:
            #print()

    return max#, lbl

def collect():
    '''src = int(input("Source:\n0 - File\n1 - Webcam\n"))'''
    name = "sample"

    global file_source,src

    # Start the video stream
    vs = VideoStream(src=0).start() if src==1 else FileVideoStream(file_source).start()

    count = 0

    while True:
        frame = vs.read()
        if frame is None:
            break

        markNcollect(frame,name,count)
        #assign name & implement count for filename
        
        # Convert the captured frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Return the frame as a bytes-like object
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        count+=1

def markNcollect(frame,name,count):
    global net

    (h, w) = frame.shape[:2]
    #blob = cv2.dnn.blobFromImage(frame, 1.0, (h, h), (104.0, 177.0, 123.0))
    #blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
    blob = cv2.dnn.blobFromImage(frame, 0.5, (h,h), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        img = frame[startY:endY,startX:endX]

        try:
            cv2.imwrite("./facedata/"+name+"/"+str(datetime.datetime.now())+str(count)+".jpg",img)
        except:
            print("imwrite error")

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/collect_feed")
def collect_feed():
    return Response(collect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    # Release the video stream
    vs.stop()

@app.route("/data")
def data():
    data = []
    try:
        with open("att_facenet.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        # Handle case where the CSV file doesn't exist
        return "CSV file not found", 404
    
    return jsonify(data)

@app.route('/filter_data')
def filter_data():
    selected_date_str = request.args.get('date')
    selected_name = request.args.get('name')
    #time_from = request.args.get('')
    #time_to = request.args.get('')

    data = []

    try:
        selected_date = datetime.datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        #time_from = 
        #time_to = 
    except:
        print('')

    with open('att_facenet.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            row_date = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f').date()
            row_name = row[1]

            if selected_date_str != '' and selected_name != '':
                if row_date == selected_date and row_name == selected_name:
                    data.append(row)
            else:
                if selected_date_str == '':
                    if row_name == selected_name: data.append(row)
                else:
                    if row_date == selected_date:
                        data.append(row)

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)