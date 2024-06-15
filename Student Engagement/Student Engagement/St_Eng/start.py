from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
import os
import cx_Oracle
from numpy import *
import warnings
warnings.filterwarnings("ignore")

app = Flask('__name__')

#face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier =load_model(r'model.h5')
person_model = load_model('personal.h5')
model = load_model(r'student.h5')


######################Registration######################################
def register_user():
    
    # Oracle database configuration
    DB_USERNAME = 'surya'
    DB_PASSWORD = 'surya'
    DB_DSN = cx_Oracle.makedsn('DESKTOP-KA1S4B6', '1521', service_name='XE')

    try:
        # Connect to Oracle database
        connection = cx_Oracle.connect(DB_USERNAME, DB_PASSWORD, DB_DSN)
        cursor = connection.cursor()

        # Example: Insert data into a table
        data_to_insert = (username,password)  # Example data to insert
        cursor.execute("INSERT INTO register (username, password) VALUES (:1, :2)", data_to_insert)

        connection.commit()
        print("Registeration Successfull")

    except cx_Oracle.DatabaseError as e:
        print("Error:", e)

        return "Unsuccessfull"

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            
    return "Registeration Successfull"


@app.route('/register_user', methods = ['GET', 'POST'])
def register_user_task():
    return render_template("register.html")

@app.route('/register_user_enter', methods = ['GET', 'POST'])
def register_user_into():

    username_u = request.form.get('username')
    password_p = request.form.get('password')

    ret_val = register_user (username_u, password_p)

    if(ret_val== "Registeration Successfull" ):
        return render_template("welcome.html")
    else:
        return render_template("register.html")


##########################Registration Close###############################

#################################password check############################
def password_check_user(u_name,pass_val):
    # Oracle database configuration
    DB_USERNAME = 'surya'
    DB_PASSWORD = 'surya'
    DB_DSN = cx_Oracle.makedsn('DESKTOP-KA1S4B6', '1521', service_name='XE')

    try:
        # Connect to Oracle database
        connection = cx_Oracle.connect(DB_USERNAME, DB_PASSWORD, DB_DSN)
        cursor = connection.cursor()

        # Example: Retrieve data from the table
        cursor.execute("SELECT username, password FROM register")
        rows = cursor.fetchall()

        if( (u_name, pass_val) in rows ):

            return True

        else:
            return False

    except cx_Oracle.DatabaseError as e:
        print("Error:", e)

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

############################################################################################

@app.route('/', methods = ['GET', 'POST'])
def instart():
    return render_template("Welcome.html")


@app.route('/password_student', methods = ['GET', 'POST'])
def password_start():
    return render_template("password_student.html")

pass_value=""


@app.route('/single_start', methods = ['GET', 'POST'])
def intro_start():

    global pass_value

    username_1 = request.form.get('username')
    password_1 = request.form.get('password')
    
    pass_value = username_1

    ret_val = password_check_user ( username_1,password_1)

    if(ret_val==True):
        return render_template("Welcome_For single detection.html")
    else:
        return "You not valid user"


###################################SINGLE IMAGE DETECTION######################################
    
engagement_labels1 = ['Apprehensive',
                     'Curious',
                     'Distracted',
                     'Enthusiastic',
                     'Irritated',
                     'Observant',
                     'Uninterested']


def pred(image_name):
    
    display=cv2.imread(image_name)
    image = cv2.imread(image_name)

  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    image = cv2.resize(image, (48,48))
    
    #expand your image array
    img=expand_dims(image,0)
    

    predictions = model.predict(img)
    # Convert the predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    emotion = engagement_labels1[predicted_labels[0]]


    print("The Engagement detected is",engagement_labels1[predicted_labels[0]])
    return emotion



def send_to_db(pass_val , value):

    # Oracle database configuration
    DB_USERNAME = 'surya'
    DB_PASSWORD = 'surya'
    DB_DSN = cx_Oracle.makedsn('DESKTOP-KA1S4B6', '1521', service_name='XE')

    try:
        # Connect to Oracle database
        connection = cx_Oracle.connect(DB_USERNAME, DB_PASSWORD, DB_DSN)
        cursor = connection.cursor()

        # Example: Insert data into a table
        data_to_insert = (pass_val,value )  # Example data to insert
        cursor.execute("INSERT INTO new (name, action) VALUES (:1, :2)", data_to_insert)

        connection.commit()
        print("Data added successfully")

    except cx_Oracle.DatabaseError as e:
        print("Error:", e)

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            
    return "Data added successfully"
                    
@app.route('/detect', methods = ['GET', 'POST'])
def upload_detection1():

    if(request.method == 'POST'):
        
        f1=request.files['file']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)

        
        file_path1 = os.path.join(
            basepath, 'uploads', secure_filename(f1.filename))

               
        f1.save(file_path1)

        value  = pred(file_path1)

        status = send_to_db(pass_value,value)

        

    
    return render_template("End_For_Single detection.html", n=value)


#########################################END############################################



#########################################LIVE DETECTION###################################
               
@app.route('/password_student_live_detection', methods = ['GET', 'POST'])
def password_start_live():
    return render_template("password_student_live_detection.html")

pass_value_live=""


@app.route('/single_start_live', methods = ['GET', 'POST'])
def intro_start_live():

    global pass_value_live

    username_2 = request.form.get('username')
    password_2 = request.form.get('password')

    pass_value_live = username

    ret_val = password_check_user ( username_2,password_2)

    if(ret_val==True):
        return render_template("start_live_capturing.html")
    else:
        return "You not valid user"
    

@app.route('/detect_live', methods = ['GET', 'POST'])
def detect_live():
    
    #engagement_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    engagement_labels = ['irritated', 'uninterested', 'apprehensive', 'enthusiastic' , 'observant', 'distracted', 'curious']


    engagement_counts = {str(label): 0 for label in range(1)}  # Adjust NUM_PERSONS based on your person model output
    engage = {str(label): "" for label in range(1)}  # Dictionary to store the most frequent engagement label for each person


    cap = cv2.VideoCapture(0)

    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1350)  # Adjust the width as per your requirement
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  

    while True: 
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            

            
            roi_gray1 = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        

            if np.sum([roi_gray1])!=0 :
        
                roi1 = roi_gray1.astype('float')/255.0
                roi1 = img_to_array(roi1)
                roi1 = np.expand_dims(roi1,axis=0)
             

                prediction = classifier.predict(roi1)[0]
                label=engagement_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
              
                
              
                engagement_label_index = prediction.argmax()
                engagement_label = engagement_labels[engagement_label_index]
              
                
                # Update engagement counts and most frequent engagement label for each person
                person_label= str(0)
                if person_label in engagement_counts:
                    engagement_counts[person_label] += 1
                    if engagement_counts[person_label] >= max(engagement_counts.values()):
                        engage[person_label] = engagement_label 
                    print(engage)
                
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the most frequent engagement label for each person
    for person, label in engage.items():
        print(f"Most frequent engagement label for person {person}: {label}")
        
    engage_value = engage['0']
    db_status=  send_to_db(pass_value_live , engage_value)

    #return db_status
    return render_template("End_For_Single detection.html", n=db_status)


@app.route('/admin_pass', methods = ['GET', 'POST'])
def admin():
    return render_template("password.html")

@app.route('/retrive_back', methods = ['GET', 'POST'])
def retrive():

    # Oracle database configuration
    DB_USERNAME = 'surya'
    DB_PASSWORD = 'surya'
    DB_DSN = cx_Oracle.makedsn('DESKTOP-KA1S4B6', '1521', service_name='XE')

    
    try:
        # Connect to Oracle database
        connection = cx_Oracle.connect(DB_USERNAME, DB_PASSWORD, DB_DSN)
        cursor = connection.cursor()

        # Example: Retrieve data from the table
        cursor.execute("SELECT name, action FROM new")
        rows = cursor.fetchall()

        # Pass the retrieved data to the template
        return render_template('Retrival_process.html', rows=rows)

    except cx_Oracle.DatabaseError as e:
        return "Error: " + str(e)

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


####################################################################################

    
if __name__ == "__main__":
    app.run(debug = False)
    
