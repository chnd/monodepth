# This file is a modification from the original file: monodepth_simple.py
# tuxvoid.blogspot.com
#
# Original copyright note:
#
# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com
#

from __future__ import division

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

# Add required modules for realtime image capture using opencv
import cv2
import multiprocessing
from utils import FPS, WebcamVideoStream                                        
from multiprocessing import Queue, Pool
import matplotlib.animation as animation

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

parser.add_argument('-src', '--source', dest='video_source', type=int, default=0, help='Device index of the camera.')          
parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int, default=2, help='Number of workers.')                   
parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int, default=5, help='Size of the queue.')                   

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params, input_image, sess):
    """Test function."""


    #output_directory = os.path.dirname(args.image_path)
    #output_name = os.path.splitext(os.path.basename(args.image_path))[0]

    #np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    #plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

    print('done!')
    return disp_to_img

def worker(input_q, output_q):                                                  
    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True                                      
                                                                                
    # SESSION
    sess = tf.Session(config=config)
                                                                                
    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path
    train_saver.restore(sess, restore_path)

    fps = FPS().start()                                                         
    while True:                                                                 
        fps.update()                                                            
        frame = input_q.get()                                                   
        if len(frame.shape) == 2:
            original_height, original_width = frame.shape
            num_channels = 1
        else:
            original_height, original_width, num_channels = frame.shape
            
        if num_channels == 4:
            frame = frame[:,:,:3]
        elif num_channels == 1:
            frame = np.tile((frame, frame, frame), 2)

        input_image = scipy.misc.imresize(frame, [args.input_height, args.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)
        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
        output_image = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
        output_image = cv2.applyColorMap(output_image, cv2.COLORMAP_JET)
        output_q.put(output_image)
                                                                                
    fps.stop()                                                                  
    sess.close()     

if __name__ == '__main__':
    logger = multiprocessing.log_to_stderr()                                    
    logger.setLevel(multiprocessing.SUBDEBUG)                                   
                                                                                
    input_q = Queue(maxsize=args.queue_size)                                    
    output_q = Queue(maxsize=args.queue_size)                                   
    pool = Pool(args.num_workers, worker, (input_q, output_q))                  
                                                                                
    video_capture = WebcamVideoStream(src=args.video_source,                    
                                      width=args.input_width,                         
                                      height=args.input_height).start()               
    fps = FPS().start()                                                         

    while True:
        frame = video_capture.read()                                            
        if (frame == None):
            continue

        input_q.put(frame)                                                      
                                                                                
        t = time.time()                                                         
                                                                                
        cv2.imshow('Original', frame)                                     
        cv2.imshow('Disparity map', output_q.get())                                     
        fps.update()                                                            
                                                                                
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))            
                                                                                
        if cv2.waitKey(1) & 0xFF == ord('q'):                                   
            break                                                               
                                                                                
    fps.stop()                                                                  
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))          
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))                       
                                                                                
    pool.terminate()                                                            
    video_capture.stop()                                                        
    cv2.destroyAllWindows()  

