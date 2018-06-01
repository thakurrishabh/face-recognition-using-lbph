import cv2
import numpy as numpy
import lbph as lr
import sys, os,time
import sys
import msvcrt


while True:
        user=input('1.train 2.recognize : ')

        if user == '1':
                # create_database.py
                #password=raw_input('Enter Admin Password : ')
                password=input('Enter Admin Password (the password is password): ')

                if password == 'Password':
                        count = 0
                        size = 4
                        fn_haar = 'haarcascade_frontalface_default.xml'
                        fn_dir = 'database'
                        fn_name = input('Enter Name Of The Person : ')   #name of the person
                        path = os.path.join(fn_dir, fn_name)
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        (im_width, im_height) = (68, 68)
                        haar_cascade = cv2.CascadeClassifier(fn_haar)
                        webcam = cv2.VideoCapture(0)


                        print("-----------------------Taking pictures----------------------")
                        print("--------------------Give some expressions---------------------")
                        # The program loops until it has 20 images of the face.

                        while count < 64:
                            (rval, im) = webcam.read()
                            im = cv2.flip(im, 1, 0)
                            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                            mini = cv2.resize(gray, ((int)(gray.shape[1] / size), (int)(gray.shape[0] / size)))
                            faces = haar_cascade.detectMultiScale(mini)
                            faces = sorted(faces, key=lambda x: x[3])
                            if faces:
                                face_i = faces[0]
                                (x, y, w, h) = [v * size for v in face_i]
                                face = gray[y:y + h, x:x + w]
                                face_resize = cv2.resize(face, (im_width, im_height))
                                pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                                       if n[0]!='.' ]+[0])[-1] + 1
                                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(im, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                                    1,(0, 255, 0))
                                time.sleep(0.38)        
                                count += 1
                           
                                
                            cv2.imshow('OpenCV', im)
                            key = cv2.waitKey(10)
                            if key == 27:
                                break
                        print(str(count) + " images taken and saved to " + fn_name +" folder in database ")
                        cv2.destroyAllWindows()
                        webcam.release()
                        
                        import sys, os,time
                        size = 4
                        fn_haar = 'haarcascade_frontalface_default.xml'
                        fn_dir = 'database'

                        print('Training...')
                        # Create a list of images and a list of corresponding names
                        (images, lables, names, id) = ([], [], {}, 0)
                        for (subdirs, dirs, files) in os.walk(fn_dir):
                            for subdir in dirs:
                                names[id] = subdir
                                subjectpath = os.path.join(fn_dir, subdir)
                                for filename in os.listdir(subjectpath):
                                    path = subjectpath + '/' + filename
                                    lable = id
                                    images.append(cv2.imread(path, 0))
                                    lables.append(int(lable))
                                id += 1
                        (im_width, im_height) = (68, 68)

                        # Create a Numpy array from the two lists above
                        (images, lables) = [numpy.array(lis) for lis in [images, lables]]
                        trained_face_recognizer=lr.train_lbph(images)
                        print('done')
                        numpy.save('trainedRec.npy',trained_face_recognizer)
                else:
                        print('Wrong Password! Authentication Failed! Try again!')

        elif user=='2':
                trained_face_recognizer=numpy.load('trainedRec.npy')
                fn_dir = 'database'

                # Load prebuilt model for Frontal Face
                cascadePath = "haarcascade_frontalface_default.xml"
                (im_width, im_height) = (68, 68)
                # Part 2: Use fisherRecognizer on camera stream
                (images, lables, names, id) = ([], [], {}, 0)
                for (subdirs, dirs, files) in os.walk(fn_dir):
                    for subdir in dirs:
                        names[id] = subdir
                        subjectpath = os.path.join(fn_dir, subdir)
                        for filename in os.listdir(subjectpath):
                            path = subjectpath + '/' + filename
                            lable = id
                            images.append(cv2.imread(path, 0))
                            lables.append(int(lable)) 
                        id += 1

                face_cascade = cv2.CascadeClassifier(cascadePath)
                webcam = cv2.VideoCapture(0)
                while True:
                    (_, im) = webcam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                        face = gray[y:y + h, x:x + w]
                        face_resize = cv2.resize(face, (im_width, im_height))
                        # Try to recognize the face
                        #inputimg = numpy.zeros((1, 4096))
                        #inputimg[0, :] = images[48, :]
                        #lr.predict_logistic_regression(inputimg, lables,trained_face_recognizer)
                        prediction=lr.predict_lbph(face_resize,trained_face_recognizer,lables)
                        #prediction = recognizer.predict(face_resize)
                        
                        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

                        if (prediction[1])<100:

                               #cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                               cv2.putText(im,'recognized - %.0f' % (prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                        else:
                          cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

                    cv2.imshow('OpenCV', im)
                    key = cv2.waitKey(10)
                    if key == 27:
                        break
                cv2.destroyAllWindows()
                webcam.release()
        else:
                print('Enter a valid input')


        
