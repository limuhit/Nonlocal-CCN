import argparse
import caffe
import cv2
import os
import numpy as np
def float2img(data):
    tdata = data * 255.0
    tdata[tdata>255] = 255.0
    tdata[tdata<0] = 0
    return tdata.transpose(1,2,0)#.astype(np.uint8)
def encoder(num, version,ch, sim=True):
    global gpu_id
    global  wk_dir
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    if sim: net = caffe.Net('./model/encoder_deploy.prototxt','./model/save/v%d/%d_sim.caffemodel'%(version,ch),caffe.TEST)
    else: net = caffe.Net('./model/encoder_deploy.prototxt','./model/save/v%d/%d_mse.caffemodel'%(version,ch),caffe.TEST)
    global base_dir
    for i in range(1,num+1):
        img =cv2.imread(os.path.join(base_dir,'%s.png'%i))
        print os.path.join(base_dir,'%s.png'%i)
        h,w = img.shape[:2]
        net.blobs['data'].reshape(1,3,h,w)
        net.blobs['data'].data[...] = img.transpose(2,0,1).astype(np.float32)
        net.forward()
        np.save('%s/%d_code.npy'%(wk_dir,i),net.blobs['encoder_out'].data[0])
        np.save('%s/%d_int.npy'%(wk_dir,i),net.blobs['encoder_int_out'].data[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--inputdir', dest = 'imagedir',
                        default='H:/image_test_set/kodak', help = 'imagedir')
    parser.add_argument('-w', '--workingdir', dest = 'wkdir',
                        default='./model/wk_tmp', help = 'wkdir')
    parser.add_argument('-n',  dest = 'num',
                        default = '24', help='image number')
    parser.add_argument('-c',  dest = 'channel',
                        default = '4', help='channel')
    parser.add_argument('-v',  dest = 'ver',
                        default = '2', help='version')
    parser.add_argument('-gpu',  dest = 'gpu',
                        default = '0', help='gpu_id')
    parser.add_argument('-sim', dest = 'sim',
                        default = '0', help = 'Choosing models trained with ms-ssim')
    parser.add_argument('-tec', dest = 'tec',
                        default = '0', help = 'Choosing models trained with ms-ssim')
    results = parser.parse_args()
    base_dir = results.imagedir
    wk_dir = results.wkdir
    gpu_id = int(results.gpu)
    ch = int(results.channel)
    num = int(results.num)
    ver = int(results.ver)
    sim = (int(results.sim)>0)
    encoder(num,ver,ch,sim)
