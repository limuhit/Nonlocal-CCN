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
def decoder(num, version,ch, sim=True, tec=True):
    global gpu_id
    global  wk_dir
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    if sim: net = caffe.Net('./model/decoder_deploy.prototxt','./model/save/v%d/%d_sim.caffemodel'%(version,ch),caffe.TEST)
    else: net = caffe.Net('./model/decoder_deploy.prototxt','./model/save/v%d/%d_mse.caffemodel'%(version,ch),caffe.TEST)
    global base_dir
    sim_net = caffe.Net('./model/sim.prototxt',caffe.TEST) 
    fmse = lambda x,y: np.average(np.square(x-y.astype(np.float)))
    fpsnr = lambda x,y: 10*np.log10(255.0*255.0/fmse(x,y))
    mse_sum = 0
    psnr_sum = 0
    sim_sum = 0
    rt = 0
    for i in range(1,num+1):
        img =cv2.imread(os.path.join(base_dir,'%s.png'%i))
        h,w = img.shape[:2]
        net.blobs['encoder_out'].reshape(1,64,h//8,w//8)
        code = np.load('%s/%d_code.npy'%(wk_dir,i))
        net.blobs['encoder_out'].data[0] = code.astype(np.float32)
        net.forward()
        gimg = float2img(net.blobs['gdata'].data[0])
        if sim:
            if tec: cv2.imwrite('./model/tmp/tec/%d_%d_our3.png'%(i,ch),gimg)
            else: cv2.imwrite('./model/tmp/kodak/%d_%d_our3.png'%(i,ch),gimg)
        else:
            if tec: cv2.imwrite('./model/tmp/tec/%d_%d_our4.png'%(i,ch),gimg)
            else: cv2.imwrite('./model/tmp/kodak/%d_%d_our4.png'%(i,ch),gimg)
        sim_net.blobs['data'].reshape(1,3,h,w)
        sim_net.blobs['data2'].reshape(1,3,h,w)
        sim_net.blobs['data'].data[...] = img.transpose(2,0,1).astype(np.float32)
        sim_net.blobs['data2'].data[...] = gimg.transpose(2,0,1).astype(np.float32)
        sim_net.forward()
        pm = fmse(img,gimg)
        pr = fpsnr(img,gimg)
        sim_sum += (sim_net.blobs['sim_loss'].data+0)
        print '%d.png, mse: %.2f, psnr: %.2f'%(i,pm,pr)
        mse_sum += pm
        psnr_sum += pr
    print 'Average mse: %.2f, psnr: %.2f, ms-sim: %.4f'%(mse_sum/float(num), psnr_sum/float(num),  sim_sum/float(num))
    if tec: prex = 'tec'
    else: prex = 'kodak'
    if sim:  f = open('./model/save/v%d/%d_%s_sim.txt'%(version,ch,prex),'w')
    else:  f = open('./model/save/v%d/%d_%s_mse.txt'%(version,ch,prex),'w')
    f.write('%.2f, %.4f'%(psnr_sum/float(num), sim_sum/float(num)))
    f.close()
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
    tec= (int(results.tec)>0)
    decoder(num,ver,ch,sim,tec)
