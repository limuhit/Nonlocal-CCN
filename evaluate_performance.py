import subprocess
import os


def compress(dir,num,ver,ch,gpu,sim,tmp_dir):
    cmd=['python', './full_encoder.py', '-d', dir, '-n', str(num), '-v', str(ver), '-c', str(ch), '-sim', str(sim), '-gpu', str(gpu), '-w', tmp_dir]
    my_env = os.environ
    my_env["PATH"] = "D:\Programs\Anaconda2;" + my_env["PATH"]
    p = subprocess.Popen(cmd, env=my_env)
    p.wait()


def decompress_rt_nlc(dir,num,ver,ch,gpu,sim,tec,tmp_dir):
    cmd=['python', './full_nlc.py', '-d', dir, '-n', str(num), '-v', str(ver), '-c', str(ch), '-sim', str(sim), '-gpu', str(gpu), '-tec', str(tec), '-w', tmp_dir]
    my_env = os.environ
    my_env["PATH"] = "D:\Programs\Anaconda2;" + my_env["PATH"]
    p = subprocess.Popen(cmd, env=my_env)
    p.wait()


def decompress(dir,num,ver,ch,gpu,sim,tec,tmp_dir):
    cmd=['python', './full_decoder.py', '-d', dir, '-n', str(num), '-v', str(ver), '-c', str(ch), '-sim', str(sim), '-gpu', str(gpu), '-tec', str(tec), '-w', tmp_dir]
    my_env = os.environ
    my_env["PATH"] = "D:\Programs\Anaconda2;" + my_env["PATH"]
    p = subprocess.Popen(cmd, env=my_env)
    p.wait()


if __name__ == '__main__':
    ch_list = [4,8,12,16,20,26,32]
    sim = 0
    ver = 2
    tec = 0
    gpu_id = 0
    tmp_dir = './model/wk_tmp'
    if not os.path.exists('./model/tmp'):
        os.mkdir('./model/tmp')
    if not os.path.exists('./model/tmp/kodak'):
        os.mkdir('./model/tmp/kodak') # directory for saving decompressed images for kodak dataset
    if not os.path.exists('./model/tmp/tec'):
        os.mkdir('./model/tmp/tec') # directory for saving decompressed images for tecnick dataset
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if tec>0:
        base_dir = 'H:/image_test_set/tec' # source image directory for tecnick dataset
        num = 100
    else:
        base_dir = 'H:/image_test_set/kodak' # source image directory for kodak dataset
        num = 24
    for ch in ch_list[:]:
        compress(base_dir,num,ver,ch,gpu_id,sim,tmp_dir)
        decompress(base_dir,num,ver,ch,gpu_id,sim,tec,tmp_dir)
        decompress_rt_nlc(base_dir,num,ver,ch,gpu_id,sim,tec,tmp_dir)

