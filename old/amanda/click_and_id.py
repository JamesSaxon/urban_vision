# import the necessary packages
import argparse
import cv2
import json
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
drawing = False

def click_and_id(event, x, y, flags, param):
    
    # grab references to the global variables
    global refPt, drawing
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that drawing is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        drawing = True
    # draw rectangle 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(image, refPt[-1], (x,y), (255, 255, 255), 2)
            cv2.imshow('image', image)
    
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the drawing operation is finished
        refPt.append((x, y))
        drawing = False

def order_bounding_box(corner1, corner2):
    upper_left = (min(corner1[0], corner2[0]), min(corner1[1], corner2[1]))
    lower_right = (max(corner1[0], corner2[0]), max(corner1[1], corner2[1]))
    return [upper_left, lower_right]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

output_dict = {'file_name': args["image"], 'image_width':None, 'image_height':None, 
               'xmins':[], 'xmaxs':[], 'ymins':[], 'ymaxs':[], 'classes':[], 
               'classes_text':[]}


 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])

output_dict['image_height']=image.shape[0]
output_dict['image_width']=image.shape[1]

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_id)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
 
    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
 
    # if the 'p' key is pressed, print coordinates, 'person', and draw rectangle in green.
    if key == ord("p"):
        bb = order_bounding_box(refPt[-2], refPt[-1)
        print('person', order_bounding_box(refPt[-2], refPt[-1]))
        output_dict['xmins'].append(bb[0][0])
        output_dict['ymins'].append(bb[0][1])
        output_dict['xmaxs'].append(bb[1][0])
        output_dict['ymaxs'].append(bb[1][1])
        output_dict['classes'].append(1)
        output_dict['classes_text'].append(b'person')
        image = clone.copy()
        cv2.rectangle(image, refPt[-2], refPt[-1], (0, 255, 0), 3)
        cv2.imshow('image', image)
        clone = image.copy()
    
    # if the 'b' key is pressed, print coordinates, 'bus', and draw rectangle in red.
    if key == ord("b"):
        bb = order_bounding_box(refPt[-2], refPt[-1)
        print('bus', order_bounding_box(refPt[-2], refPt[-1]))
        output_dict['xmins'].append(bb[0][0])
        output_dict['ymins'].append(bb[0][1])
        output_dict['xmaxs'].append(bb[1][0])
        output_dict['ymaxs'].append(bb[1][1])
        output_dict['classes'].append(2)
        output_dict['classes_text'].append(b'bus')
        image = clone.copy()
        cv2.rectangle(image, refPt[-2], refPt[-1], (0, 0, 255), 3)
        cv2.imshow('image', image)
        clone = image.copy()

    # if the 'c' key is pressed, print coordinates, 'car', and draw rectangle in yellow.
    if key == ord("c"):
        bb = order_bounding_box(refPt[-2], refPt[-1)
        print('person', order_bounding_box(refPt[-2], refPt[-1]))
        output_dict['xmins'].append(bb[0][0])
        output_dict['ymins'].append(bb[0][1])
        output_dict['xmaxs'].append(bb[1][0])
        output_dict['ymaxs'].append(bb[1][1])
        output_dict['classes'].append(3)
        output_dict['classes_text'].append(b'car')
        image = clone.copy()
        cv2.rectangle(image, refPt[-2], refPt[-1], (0, 255, 255), 3)
        cv2.imshow('image', image)
        clone = image.copy()

    # if the 'q' key is pressed, break from the loop
    elif key == ord("q"):
        break

#Write output_dict to file
with open('output.json', 'w') as fp:
    json.dump(output_dict, fp)
 
# close all open windows
cv2.destroyAllWindows()
