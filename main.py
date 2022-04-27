import os
import mahotas
import numpy
from pylab import gray, imshow, show
import numpy as np
import cv2
import imutils
import re
import joblib
import time
from collections import defaultdict


def main(paths):
    processed = []
    """computes image moments in the order of blue, green, red, greyscale"""
    img_moments = []
    try:
        show_image = False
        for img_path in paths:
            img = cv2.imread(img_path) #reads as BGR
            name = img_path.split('/')[1]
            moments = [name]
            for i, color in enumerate(['b', 'g', 'r']): # BGR
                channel = np.zeros(img.shape)
                channel[:, :, i] = img[:, :, i]
                cv2.imwrite(f"{i}.jpg", channel)
                try:
                    moments.append(describe_image(channel, show_image))
                except Exception as e:
                    print("exception\n", e)
                    print(img_path, i)
            moments.append(describe_image(img, show_image))
            img_moments.append(moments)
            processed.append(img_path[4:])
            if(len(processed)>20):
                with open('processed.txt', 'a') as file:
                    file.write(''.join([img+"\n" for img in processed]))
                processed=[]
    except Exception as e:
        print(e)
    # delete any rgb images that were created
    finally:
        with open('processed.txt', 'a') as file:
            file.write(''.join([img+"\n" for img in processed]))
        return img_moments



def describe_image(img, show_image):
    # blur and convert to greyscale
    img = np.float32(img)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    image_grey = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # threshold image. object to be found should be white and background should be black
    avg = np.mean(image_grey)
    bright_pixels = np.count_nonzero(image_grey > avg)
    dark_pixels = np.count_nonzero(image_grey <= avg)
    threshold=0
    if(bright_pixels < dark_pixels):
        threshold = cv2.threshold(image_grey, avg, 255, cv2.THRESH_BINARY)[1]
    else:
        threshold = cv2.threshold(
            image_grey, avg, 255, cv2.THRESH_BINARY_INV)[1]
    threshold = cv2.dilate(threshold, None, iterations=4)
    threshold = cv2.erode(threshold, None, iterations=2)
    threshold = np.uint8(threshold)

    # find contour
    outline = np.zeros(shape=image_grey.shape, dtype="uint8")
    # cv2.RETR_EXTERNAL finds only the outermost contours
    # cv2.CHAIN_APPROX_SIMPLE compresses and approximates the contours to save memory
    contours = cv2.findContours(
        threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(outline, [largest_contour], -1, 255, -1)
    
    # dynamically compute radius such that it returns the min. radius that enclose the entire object
    return mahotas.features.zernike_moments(outline, cv2.minEnclosingCircle(largest_contour)[1], degree=8, cm=None)


def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #get list of processed images
    proc_img = []
    with open('processed.txt', 'r') as file:
        proc_img = set([img.strip() for img in file.readlines()])
    #get all images that haven't been unprocessed yet
    unproc_img = ["img/"+i for i in os.listdir("./img/") if i not in proc_img]
    print("number of processed images:",len(proc_img))
    print("number of unprocessed images:",len(unproc_img), "\n")

    #parallel computing
    cpu_num = joblib.cpu_count()
    batches = list(numpy.array_split(unproc_img, cpu_num))
    #create a list of delayed function calls to main() without actually executing
    delayed_funcs = []
    for batch in batches:
        delayed_funcs.append(joblib.delayed(main)(batch))
    start = time.time()
    #executes the list of delayed functions simultaneously
    #img_moments stores all the returned values for all the threads. thus, len(all_moments) == len(unproc_img)
    all_moments = joblib.Parallel(n_jobs = cpu_num, verbose = 10)(delayed_funcs)
    # all_moments is a list of tuples of image moments: 
    # [ (file_name, Blue channel moments, Green channel moments, Red channel moments, total moments) ]
    end = time.time()
    descriptors = []
    for each in all_moments:
        descriptors.extend(each)
    print("\nnumber of processed images =",len(descriptors))
    print("number of descriptors for each moment =",len(descriptors[0][1]))
    for d in descriptors:
        print("\n\t\tmoments for image:", d[0])
        print("blue channel moments =", d[1])
        print("green channel moments =", d[2])
        print("red channel moments =", d[3])
        print("total channel moments =", d[4])
        break
    print("time took:", end-start)
    with open('processed.txt', 'w') as file:
        file.close()
