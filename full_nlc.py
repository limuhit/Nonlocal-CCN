import argparse
import caffe
import cv2
import os
import numpy as np
def decoder(num, ch, sim=True, tec=True):
    global gpu_id
    global wk_dir
    global ver
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    pre = caffe.Net('./model/nlc/%d_patch.prototxt'%ch,caffe.TEST)
    post = caffe.Net('./model/nlc/%d_depatch.prototxt'%ch,caffe.TEST)
    loss =  caffe.Net('./model/nlc/%d_loss.prototxt'%ch,caffe.TEST)
    if sim: 
        net = caffe.Net('./model/nlc/%d_nlc.prototxt'%ch,'./model/save/v%d/%d_sim_nlc.caffemodel'%(ver,ch),caffe.TEST)
        #net = caffe.Net('./model/nlc/%d_nlc.prototxt'%ch,'./model/save/nlc_26_iter_250000.caffemodel',caffe.TEST)
    else: 
        net = caffe.Net('./model/nlc/%d_nlc.prototxt'%ch,'./model/save/v%d/%d_mse_nlc.caffemodel'%(ver,ch),caffe.TEST)
        #net = caffe.Net('./model/nlc/%d_nlc.prototxt'%ch,'./model/save/nlc_4_iter_210000.caffemodel',caffe.TEST)
    rt = 0
    for idx in range(1,num+1):
        int_code = np.load('%s/%d_int.npy'%(wk_dir,idx))
        #print int_code.shape
        pre.blobs['data'].reshape(1,ch,int_code.shape[1],int_code.shape[2])
        pre.blobs['data'].data[0] = int_code[:ch]
        pre.forward()
        loss.blobs['data'].reshape(1,ch,int_code.shape[1],int_code.shape[2])
        loss.blobs['data4'].reshape(1,ch*8,int_code.shape[1],int_code.shape[2])
        loss.reshape()
        post.blobs['shape'].data[...]=pre.blobs['shape'].data
        tnum = pre.blobs['patch'].shape[0]
        post.blobs['data3'].reshape(tnum,net.blobs['pdata'].shape[1],pre.blobs['patch'].shape[2],pre.blobs['patch'].shape[3])
        post.reshape()
        net.blobs['data2'].reshape(1,ch,pre.blobs['patch'].shape[2],pre.blobs['patch'].shape[3])
        for i in range(tnum):
            net.blobs['data2'].data[0] = pre.blobs['patch'].data[i]
            net.forward()
            post.blobs['data3'].data[i] = np.copy(net.blobs['pdata'].data[0])
        post.forward()
        loss.blobs['data'].data[0] = int_code[:ch]
        loss.blobs['data4'].data[...] = post.blobs['pdata_depatch'].data
        loss.forward()
        rt += (loss.blobs['ent_loss'].data+0)
        print '%d.png, entropy: %.3f'%(idx,loss.blobs['ent_loss'].data+0)
        bpp = (loss.blobs['ent_loss'].data+0)/0.693*ch/64
        if sim:
            if tec:  os.rename('./model/tmp/tec/%d_%d_our3.png'%(idx,ch),'./model/tmp/tec/%d_%d_our3.png'%(idx,int(bpp*1000)))
            else: os.rename('./model/tmp/kodak/%d_%d_our3.png'%(idx,ch),'./model/tmp/kodak/%d_%d_our3.png'%(idx,int(bpp*1000)))
        else:
            if tec:  os.rename('./model/tmp/tec/%d_%d_our4.png'%(idx,ch),'./model/tmp/tec/%d_%d_our4.png'%(idx,int(bpp*1000)))
            else: os.rename('./model/tmp/kodak/%d_%d_our4.png'%(idx,ch),'./model/tmp/kodak/%d_%d_our4.png'%(idx,int(bpp*1000)))
    print 'Average entropy: %.3f'%(rt/float(num))
    print  '%.3f'%(rt/float(num)/0.693*ch/64)
    if tec: prex = 'tec'
    else: prex = 'kodak'
    if sim:  fname = './model/save/v%d/%d_%s_sim.txt'%(ver,ch,prex)
    else:  fname = './model/save/v%d/%d_%s_mse.txt'%(ver,ch,prex)
    f = open(fname)
    old_content = f.readline()
    f.close()
    f = open(fname,'w')
    f.write('%.3f, %s'%(rt/float(num)/0.693*ch/64,old_content))
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
    tec = (int(results.tec)>0)
    decoder(num, ch,sim,tec)
