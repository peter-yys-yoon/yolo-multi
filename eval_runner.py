import os
import time
import argparse
PY = 'test.py'
EXE ='/home/peter/anaconda3/envs/eryolo/bin/python'

#CMD = f'{EXE} {PY} --data_config {} --iou_thres {} --conf_thres {} --weights_path {}'


parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=0, help='device id')
opt = parser.parse_args()

#config_set = [ ('./config/tang_crop3.data', './checkpoints/scratch_crop3') ]
#config_set = [ ('./config/tang_crop3.data', './checkpoints/pre_crop3') ]
config_set = [ ('./config/tang_crop3.data', './checkpoints/pre_crop3_ep160') ]
iou_list=[0.45, 0.65]
nms_list=[0.45,0.65]

for config in config_set:
    config_file, weight_dir = config
    wlist = os.listdir(weight_dir)
    wlist.sort()
    wlist.reverse()
    nwlist = []
    for ww in wlist:
        tmp = ww.split('.')[0]
        hidx = tmp.index('h')
        #ep = int(ww.split('.')[0][-2:])
        ep = int(tmp[hidx+1:])
        if ep > 15 and ep% 4 == 0:
            nwlist.append(ww)

            #print(ep)
    #quit()
    step = int(len(nwlist)/4)
    nnwlist = nwlist[opt.id*step:(opt.id+1)*step]
    #quit()

    for weight_name in nnwlist:
        wpath = os.path.join(weight_dir,weight_name)
        print(config_file, wpath)

        for iou_thres in iou_list:
            for nms_thres in nms_list:
                ckpt = time.time()
                CMD = f'CUDA_VISIBLE_DEVICES={opt.id} {EXE} {PY} --data_config {config_file} --nms_thres {nms_thres} --iou_thres {iou_thres} --weights_path {wpath}'
                os.system(CMD)
                print('----------> took', time.time() -ckpt)
                #quit()
