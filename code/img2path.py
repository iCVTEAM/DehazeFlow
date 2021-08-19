import glob
from natsort import natsort
import numpy as np
import os


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def its():

    lr_dir = 'D:/datasets/RESIDE/hazy'
    hr_dir = 'D:/datasets/RESIDE/gt'

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))

    indices = np.arange(13990)
    np.random.shuffle(indices)

    lr_paths = [lr_paths[i] for i in indices[:13990]]

    with open("../datasets/its_hazy_train.txt", "w") as f:
        for i in range(13990):
            f.write(os.path.join(lr_dir,lr_paths[i])+'\n')
    with open("../datasets/its_clear_train.txt", "w") as f:
        for i in range(13990):
            f.write(os.path.join(hr_dir,lr_paths[i].split('/')[-1].split('_')[0]+'.png')+'\n')


def ots():
    lr_dir = 'D:\BaiduNetdiskDownload\OTS\hazy'
    hr_dir = 'D:\BaiduNetdiskDownload\OTS\gt'


    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))

    indices = np.arange(313950)
    np.random.shuffle(indices)

    lr_paths = [lr_paths[i] for i in indices[:313950]]

    with open("../datasets/its_hazy_val.txt", "w") as f:
        for i in range(len(lr_paths)):
            f.write(os.path.join(lr_dir,lr_paths[i])+'\n')
    with open("../datasets/its_clear_val.txt", "w") as f:
        for i in range(len(lr_paths)):
            f.write(os.path.join(hr_dir,lr_paths[i].split('/')[-1].split('_')[0]+'.png')+'\n')



if __name__ == "__main__":
    its()
    ots()
