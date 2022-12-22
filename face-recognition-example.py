import face_recognition
import os 
import cv2

KNOWN_FACES_DIR = 'known_faces'
UNKNOWN_FACES_DIR = 'unknown_faces'

TOLERANCE = 0.6

FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn' # hog

print('\nloading known faces. . .\n')

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):
        img = cv2.imread(f'{KNOWN_FACES_DIR}/{name}/{filename}')
        dim = (120,80)
        resized_image = cv2.resize(img,dim,cv2.INTER_AREA)

        image = face_recognition.load_image_file(resized_image)
        
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print('\nprocessing uknonwn faces. . .\n')

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(f'Filename {filename}', end='')
    original_image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

    img = cv2.imread(original_image)
    dim = (120,80)
    image = cv2.resize(img,dim,cv2.INTER_AREA)

    loactions = face_recognition.face_locations(image, model = MODEL)
    encoding = face_recognition.face_encodings(image, loactions)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoding, loactions):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None# match = known_faces[results.index(True)]
        
        if True in results:
            match = known_faces[results.index(True)]
            print(f'Match Found . . : {match}')

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [255, 0, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (150, 150, 150), FONT_THICKNESS)

    cv2.imshow(filename, image)
    # cv2.waitKey(0)# --1
    # cv2.destroyWindow(filename)# --2
    # --1 and 2
    cv2.waitKey(1000)