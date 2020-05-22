import io
import time
import threading
import picamera
import numpy as np
from PIL import Image, ImageFile
import random
import cv2 
import datetime
import os
import RPi.GPIO as GPIO
from keras.models import load_model
import pytesseract
from matplotlib import pyplot as plt
import math
from queue import Queue
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from multiprocessing import Process
import socket




threads=[]
q=Queue(maxsize=0)
session = tf.Session()
graph=tf.get_default_graph()
set_session(session)
model = load_model('tag_model_v2.h5')
data=0
class Tag_detect(threading.Thread):
    def __init__(self,queue,cnt):
        super(Tag_detect, self).__init__()
        self.cnt=0

    def tag_read(self,image):
        global session
        global graph
        set_session(session)
        image = image.reshape(1,78,78,1)
        image = image.astype('float32')
        image /= 255
        with graph.as_default():
            prediction = model.predict(image,batch_size=1,verbose=1)
            pred_class = np.argmax(prediction,axis = 1)
            pred_class = pred_class
            return pred_class


    def tag_extract(self,image):
        print("in tag extraction")
        img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, (30, 40, 20), (90, 255,255))
        
        imask = mask>0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)

        # detect circles
        detected_circles = cv2.HoughCircles(gray,  
                           cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                       param2 = 1, minRadius = 10, maxRadius = 50) 

        if detected_circles is not None: 

            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles))
            #print(detected_circles)
            pt = detected_circles[0,0]
            # a: Vertical coord;   b: horizontal coord
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(green, (a, b), r, (255, 255, 255), 2) 
        else:
            return None
        # crop the tag
        roi = img[b-r:b+r,a-r:a+r]
        
        # empty image if tag is on the edge
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            tag = np.zeros((78,78))
            tag = np.uint8(tag)
            return tag
        
        # make tag a scare
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if gray_roi.shape[0] != gray_roi.shape[1]:
            if gray_roi.shape[0] < gray_roi.shape[1]:
                new_roi = np.zeros((gray_roi.shape[1],gray_roi.shape[1]))
                new_roi[0:gray_roi.shape[0],0:gray_roi.shape[1]] = gray_roi
            else: 
                new_roi = np.zeros((gray_roi.shape[0],gray_roi.shape[0]))
                new_roi[0:gray_roi.shape[0],0:gray_roi.shape[1]] = gray_roi
            gray_roi = new_roi

        # crop to size 78*78
        if gray_roi.shape[0] > 78:
            gray_roi = gray_roi[r-39:r+39,r-39:r+39]
            tag = gray_roi
        elif roi.shape[0] < 78:
            tag = np.zeros((78,78))
            tag[39-r:39+r,39-r:39+r] = gray_roi
            tag = np.uint8(tag)
        else:
            tag = gray_roi
        return tag
       

    def run(self):
        global a
        with lock:
            #print ("in tag"+str(self.cnt))
            #print (self.image)
            f=open('data.txt',"a")
            #image=np.array(q.get().convert('RGB'))
            image,self.cnt=q.get()
            
            
            tag=self.tag_extract(np.array(image))
            
            pred=self.tag_read(tag)
            #cv2.imwrite('/home/pi/Desktop/4/'+str(self.cnt)+'.jpg',tag)
            #print("pred: "+str(pred))
            if tag is not None:
            #    print('detect')
            #pred=self.tag_read(tag)
                print (pred)
                f.write(str(self.cnt)+'   '+str(pred)+" %4d \n"%(time.time()-a))
            else:
                f.write(str(self.cnt)+'   '+"None %4d \n"%(time.time()-a))
            
    
    

# Create a pool of image processors
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)

DATE_FMT_STR='%Y-%m-%d_%H-%M-%S'
DATE_FMT_STR_IMG='%Y-%m-%d_%H-%M-%S_%f'

def get_run_count(runFile='runCount.dat'):
    '''
    Get the current run count, increment, and return
    '''
    fh=open(runFile,'r')
    s=fh.readline()
    fh.close()
    count=int(s)
    fh=open(runFile,'w')
    fh.write(str(count+1) + '\n')
    
    fh.close()
    return count

def format_folder(dt):
    '''
    Create folder and return save_prefix
    '''
    usb_dir=''
    #Check if a usb key is mounted
    if not os.system('lsblk | grep sda1'):
        usb_dir = '/media/pi/'
    else:
        print('WARNING: Did not find usb-key, writing to local dir.')
        
    pref=dt.strftime('_' + DATE_FMT_STR)
    pref = usb_dir + str(get_run_count()) +pref 
    os.mkdir(pref)
    os.mkdir(pref + '/var')
    return pref + '/'


save_prefix=format_folder(datetime.datetime.today())
print (save_prefix)

##

class ImageProcessor(threading.Thread):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.start()
        self.cnt=0
        
    def run(self):
        # This method runs in a separate thread
        GPIO.output(26, 1)
        global cnt
        global test
        while not self.terminated:
            
            if self.event.wait(1):
                try:
                    flag=0
                    self.stream.seek(0)
                    ImageFile.LOAD_TRUNCATED_IMAGES=True
                    a=Image.open(self.stream)
                    origin=a.crop((7,130,640,480))
                    # f=origin.convert('LA')
                    
                    # #print (self.cnt)
                    # threshold = 30
                    # im = f.point(lambda p: p > threshold and 255)  
               
                    # fp=np.uint16(f)/255
               
                    # pxl_count=sum(sum(fp))
                    # print(str(self.cnt) + ' '+str(pxl_count))
                    # if pxl_count[0] <52500:
                        # flag=1
                   # a.save(save_prefix+str(self.cnt)+"_%d"%flag+'.jpg')
                    #if not flag:
                    
                        
                    #name=save_prefix+'var/'+str(self.cnt)+"_%d"%flag+'.jpg'
                    
                    
                    
                    
                    #cv2.imwrite(save_prefix+'var/'+str(self.cnt)+'.png',np.array(im.convert('RGB') ))
                    #if flag:
                    q.put((origin,self.cnt))
                    threads.append(Tag_detect(q,self.cnt))
                    
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the pool
                    with lock:
                        pool.append(self)




def streams():
    
    global done 
    global cnt
    global test
    b=0
    global a
    c=a
    global data
    while not done:
        #print("in threading: ",str(data))

        with lock:
            if data==1:
                if pool:
                    processor = pool.pop()
            else:
                processor = None
                if threads:
                    threads.pop().start()
            
        if processor:
            if cnt %50==0:
                if b:
                    c=b
                b=time.time()
                #print ('cnt',str(cnt))
                print('Captured %d frames at %.2ffps' % (50,50 / (b - c)))
            cnt+=1
            processor.cnt=cnt
            yield processor.stream
            processor.event.set()
    
        else:
            # When the pool is starved, wait a while for it to refill
            time.sleep(0.1)



        



done= False
lock = threading.Lock()
pool = []
cnt = 0
test=0



'''Dictionary for bee enter/exit events'''
bee_log_dict={-1: {'entries':[], 'exits': []} }
bee_time={}
start_time=time.time()
start_time=time.time()
bee_log_dict['start_time']=start_time
bee_log_dict['start_time_iso']=datetime.datetime.today().isoformat()

#camera setup

camera=picamera.PiCamera()
camera.resolution = (640, 480)
camera.iso = 800
camera.shutter_speed=2000
camera.framerate = 45
pool = [ImageProcessor() for i in range(20)]


#GPIO setup
def GPIO13_callback(channel):
    global done
    done =True
    print("exit")
    while pool:
        with lock:
            processor = pool.pop()
        processor.terminated = True
        processor.join()
    GPIO.output(26, 0)
    camera.close()                   
    GPIO.cleanup()
    f.close()
    
GPIO.setup(13, GPIO.IN)   #bailout button
GPIO.add_event_detect(13, GPIO.RISING, callback=GPIO13_callback, bouncetime=300)

time.sleep(2)
terminate_udp=0
def udp_thread():
    UDP_IP="169.254.35.15"
    UDP_PORT=5005
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind((UDP_IP,UDP_PORT))
    global data
    global terminate_udp
    while not terminate_udp:
        d,addr =sock.recvfrom(1024)
        d=d.decode()
        #print("time diff: "+str(time.time()-float(d[2:])))
        data=int(d[0])
        if(data==1 or data==0):
            print("received :",data)
        
        
        
udp=threading.Thread(target=udp_thread,args=()).start()
while True:
    if int(data)==1:
        global a
        a=time.time()
        #camera.start_preview()
        camera.capture_sequence(streams(), use_video_port=True)
        #camera.stop_preview()
        b=time.time()
        frames=cnt
        print('Captured %d frames at %.2ffps' % (
        frames,
        frames / (b - a)))
        break

    






terminate_udp=1


udp.join()
udp.end()

# Shut down the processors in an orderly fashion


while pool:
    with lock:
        processor = pool.pop()
    processor.terminated = True
    processor.join()

GPIO.cleanup()
f.close()
