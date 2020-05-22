import socket
import time

import io
import time
import threading
import picamera
import numpy as np
from PIL import Image

import cv2 
import datetime
import os
import RPi.GPIO as GPIO
from matplotlib import pyplot as plt
import math


#set up UDP
#def UDP(
UDP_IP="169.254.35.15"
UDP_PORT=5005
	#MESSAGE = str(message).encode() 
	# print ("UDP target IP:", UDP_IP)
	# print ("UDP target port:", UDP_PORT)
	#print ("message:", MESSAGE)
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
#sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))


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
        usb_dir = '/media/pi/98DA-5580/'
    else:
        print('WARNING: Did not find usb-key, writing to local dir.')
        
    pref=dt.strftime('__' + DATE_FMT_STR)
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
        global top_camera
        global last_time_bee_enter
        while not self.terminated:
            
            if self.event.wait(1):
                try:
                    flag=0
                    self.stream.seek(0)
                    origin=Image.open(self.stream).crop((7,130,640,350))
                    f=origin.convert('LA')
                    
                    #print (self.cnt)
                    threshold = 30
                    im = f.point(lambda p: p > threshold and 255)  
               
                    fp=np.uint16(f)/255
               
                    pxl_count=sum(sum(fp))
                    #print(str(self.cnt) + ' '+str(pxl_count))
                    if pxl_count[0]<15000:
                        flag=1
		        
                        if top_camera==0:
                            top_camera=1
                            message="1 %2d"%time.time()
                            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                            print(message)
                            #UDP("1 " + str(self.cnt)) #signal top camera
                        else:
                            message="1 %2d"%time.time()
                            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                        last_time_bee_enter=time.time()
                    else:
                        if top_camera==1 and time.time()-last_time_bee_enter>3:
                            #UDP("0 " + str(self.cnt))
                            message="0 %2d"%time.time()
                            print(message)
                            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                            top_camera=0
                        else:
                            message="2 %2d"%time.time()
                            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                    #origin.save(save_prefix+str(self.cnt)+"_%d"%flag+'.jpg')
                    #if not flag:
                    
                        
                    #name=save_prefix+'var/'+str(self.cnt)+"_%d"%flag+'.jpg'
                    #cv2.imwrite(save_prefix+'var/'+str(self.cnt)+'_%2d.png'%flag,np.array(im.convert('RGB') ))
                    
                        
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
    while not done:
        with lock:
            
            if pool:
                processor = pool.pop()
            else:
                processor = None
                
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



        

#flag to indicate the top camera to start
top_camera=0
last_time_bee_enter=0

done= False
lock = threading.Lock()
pool = []
cnt = 0
test=0





'''Dictionary for bee enter/exit events'''
bee_log_dict={-1: {'entries':[], 'exits': []} }
bee_time={}

start_time=time.time()
bee_log_dict['start_time']=start_time
bee_log_dict['start_time_iso']=datetime.datetime.today().isoformat()

camera=picamera.PiCamera()
pool = [ImageProcessor() for i in range(10)]
camera.resolution = (640, 480)
camera.iso = 800
camera.shutter_speed=2000
camera.framerate = 30
time.sleep(2)

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



global a
a=time.time()
#camera.start_preview()
camera.capture_sequence(streams(), use_video_port=True)
#camera.stop_preview()
b=time.time()
frames=400
print('Captured %d frames at %.2ffps' % (
cnt,
cnnt / (b - a)))

# Shut down the processors in an orderly fashion


while pool:
    with lock:
        processor = pool.pop()
    processor.terminated = True
    processor.join()

GPIO.cleanup()

