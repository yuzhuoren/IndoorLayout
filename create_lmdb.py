

caffe_root = '/home/eeb439/caffe-future/'
import sys
sys.path.insert(0, '/home/eeb439/caffe-future/python/')

import caffe
import lmdb
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

caffe.set_mode_gpu()
caffe.set_device(1)

def create_lmdb(input_dir, save_dir, keyword, options):

    DEBUG = False

    images = glob.glob(input_dir + '*.jpg')
    #images = images[0:9]
    random.shuffle(images)

    # open the dbs
    im_db = lmdb.open(save_dir + '/' + keyword + '-image-lmdb', map_size=int(1e12))
    gc_db = lmdb.open(save_dir + '/' + keyword + '-gc_label-lmdb', map_size=int(1e12))
    bb_db = lmdb.open(save_dir + '/' + keyword + '-bb_label-lmdb', map_size=int(1e12))

    # create the image db
    with im_db.begin(write=True) as im_txn:
        for im_idx, im_ in enumerate(images):

            #if im_idx % 100 == 0:
            #    print keyword, 'image', im_idx

            print keyword, "image: ", im_idx, "/", len(images), im_

            im = caffe.io.load_image(im_).astype('float') # or load whatever numpy ndarray you need

            # convert image to required format RGB -> BGR
            bgr_im = np.zeros(shape=im.shape, dtype=np.float);
            bgr_im[:, :, 0] = im[:, :, 2]; # B
            bgr_im[:, :, 1] = im[:, :, 1]; # G
            bgr_im[:, :, 2] = im[:, :, 0]; # R
            im = bgr_im

            if DEBUG:
                plt.figure()
                plt.imshow(bgr_im)
                plt.figure()
                plt.imshow(im)
                plt.show()
                raw_input("Press Enter to continue...")

            im_dat = caffe.io.array_to_datum(im.transpose((2, 0, 1)))
            im_txn.put('{:0>10d}'.format(im_idx), im_dat.SerializeToString())

    im_db.close()

    # create the gc label db
    with gc_db.begin(write=True) as lb_txn:
        for im_idx, im_ in enumerate(images):

            print keyword, "GC label: ", im_idx, "/", len(images), im_

            # load label and convert to required format
            lb = np.loadtxt(im_[:-4] + '_gc.txt', delimiter = ',')

            if options['label-type'] == 'numeric':
                lb_dat = np.reshape(lb, (1, lb.shape[0], lb.shape[1]))
            elif options['label-type'] == 'binary':
                # load and convert label to binary
                lb_dat = np.zeros((6, lb.shape[0], lb.shape[1]))
                for i in range(0, lb.shape[0]):
                    for j in range(0, lb.shape[1]):
                        lb_dat[lb[i, j], i, j] = 1

            lb_dat = caffe.io.array_to_datum(lb_dat)
            lb_txn.put('{:0>10d}'.format(im_idx), lb_dat.SerializeToString())

    gc_db.close()

    # create the bb label db
    with bb_db.begin(write=True) as lb_txn:
        for im_idx, im_ in enumerate(images):

            print keyword, "BB label: ", im_idx, "/", len(images), im_

            # load label and convert to required format
            lb = np.loadtxt(im_[:-4] + '_bb.txt', delimiter = ',')

            if int(options['parts']) == 1: # make everything that is greater than 0 = 1
                lb[lb > 0] = 1

            if options['label-type'] == 'numeric':
                lb_dat = np.reshape(lb, (1, lb.shape[0], lb.shape[1]))
            elif options['label-type'] == 'binary':
                # load and convert label to binary
                lb_dat = np.zeros((2, lb.shape[0], lb.shape[1]))
                for i in range(0, lb.shape[0]):
                    for j in range(0, lb.shape[1]):
                        lb_dat[lb[i, j], i, j] = 1

            lb_dat = caffe.io.array_to_datum(lb_dat)
            lb_txn.put('{:0>10d}'.format(im_idx), lb_dat.SerializeToString())

    bb_db.close()

    num_samples = len(images)
    return num_samples

# create_lmdb ends


if __name__ == '__main__':

    random.seed(1)

    if len(sys.argv) < 2:
        print "python create_lmdb_BB_GC.py <#edge classes = 1 or 3>"

    parts = sys.argv[1]
    print "Using " + parts + " parts"

    dataset = 'LSUN'

    train_dir = '/media/eeb439/19df5a04-d41d-41da-a3f2-54c0bac9bae9/eeb439/Documents/Yuzhuo/LSUN_2016/size_404_joint_train_dil_7_correct_surface/' + dataset +'/train/'
    val_dir   = '/media/eeb439/19df5a04-d41d-41da-a3f2-54c0bac9bae9/eeb439/Documents/Yuzhuo/LSUN_2016/size_404_joint_train_dil_7_correct_surface/' + dataset +'/val/'
    test_dir  = '/media/eeb439/19df5a04-d41d-41da-a3f2-54c0bac9bae9/eeb439/Documents/Yuzhuo/LSUN_2016/size_404_joint_train_dil_7_correct_surface/' + dataset +'/test/'
    save_dir  = '/media/eeb439/19df5a04-d41d-41da-a3f2-54c0bac9bae9/eeb439/Documents/Yuzhuo/LSUN_2016/size_404_joint_train_dil_7_correct_surface/' + dataset +'/lmdb_parts-' + parts +'/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    options = {}
    options['label-type'] = 'numeric' # other option is binary
    options['parts']      = parts
    #import pdb
    #pdb.set_trace()
    num_train = create_lmdb(train_dir, save_dir, 'train', options)
    num_val   = create_lmdb(val_dir, save_dir, 'val', options)
    num_test  = create_lmdb(test_dir, save_dir, 'test', options)

    print '\n\nLMDB with %d train, %d val and %d test samples created.\n'%(num_train, num_val, num_test)
