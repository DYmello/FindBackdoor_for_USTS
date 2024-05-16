from __future__ import print_function
import os, sys, time
import multiprocessing as mp
import random
import csv
from collections import OrderedDict
import pickle

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

class AnnotateWorker:
    def __init__(self, anno_path, im_src, im_dst):
        self.anno_path = anno_path
        self.im_src = im_src
        self.im_dst = im_dst

    def __call__(self, args):
        i, (im_path, anno) = args
        with open(os.path.join(self.anno_path, '%07d.txt' % i), 'w') as f:
            f.write('\n'.join([','.join(map(str, x)) for x in anno]))

        ext = im_path.split('.')[-1]
        src = os.path.join(self.im_src, im_path)
        dst = os.path.join(self.im_dst, '%07d.%s' % (i, ext))
        os.system('ln -s -f %s %s' % (src, dst))

if __name__ == '__main__':
    # multiprocessing workers
    p = mp.Pool(8)

    ############################################################
    #  Choose only 'warning', 'speedlimit' and 'stop' superclasses
    ############################################################
    print_flush('Filtering raw dataset', end=' ... ')
    t1 = time.time()

    categories = \
    """warning:addedLane,curveLeft,curveRight,dip,intersection,laneEnds,merge,pedestrianCrossing,roundAbout,signalAhead,slow,speedBumpsAhead,stopAhead,thruMergeLeft,thruMergeRight,turnLeft,turnRight,yieldAhead,warningUrdbl
    speedLimit:speedLimit15,speedLimit25,speedLimit30,speedLimit35,speedLimit40,speedLimit45,speedLimit50,speedLimit55,speedLimit65,speedLimitUrdbl
    stop:stop"""

    categories = {k.split(':')[0].strip().lower(): [tag.strip().lower() for tag in k.split(':')[1].split(',')] for k in categories.split('\n')}
    inv_categories = {}
    for k, v in categories.items():
        for c in v:
            inv_categories[c] = k.strip().lower()

    allAnnotations = []
    header = open('/Users/dymello/usts/raw/allAnnotations.csv', 'r').readline()
    header = header.strip().split(';')

    class_stat = {c: 0 for c in categories.keys()}

    # Parse annotations
    with open('/Users/dymello/usts/raw/allAnnotations.csv') as csvfile_trn:
        csv_reader = csv.DictReader(csvfile_trn, delimiter=';')
        for row in csv_reader:
            for clss in class_stat.keys():
                if row['Annotation tag'].lower() in categories[clss]:
                    allAnnotations.append(row)
                    class_stat[clss] += 1

    with open('/Users/dymello/usts/raw/allFiltered.csv', 'w') as csvfile_all:
        csv_writer = csv.DictWriter(csvfile_all, fieldnames=header, delimiter=';')
        csv_writer.writeheader()
        for row in allAnnotations:
            csv_writer.writerow(row)

    print_flush('Done.')
    print_flush('Filtered dataset statistics: %s' % class_stat)
    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))

    ############################################################
    #  Extract annotations to folder ./Annotations
    #  Create soft links to all samples in folder ./Images
    ############################################################
    print_flush('Extracting annotations', end=' ... ')
    t1 = time.time()

    if not os.path.exists('/Users/dymello/usts/Annotations'):
        os.mkdir('/Users/dymello/usts/Annotations')
    if not os.path.exists('/Users/dymello/usts/Images'):
        os.mkdir('/Users/dymello/usts/Images')

    images_dict = OrderedDict()
    for row in allAnnotations:
        clss = (inv_categories[row['Annotation tag'].lower()],)
        bbox = tuple(int(row[k]) for k in ['Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y'])
        cmmt = ('clean',)
        if row['Filename'] not in images_dict:
            images_dict[row['Filename']] = [clss + bbox + cmmt]
        else:
            images_dict[row['Filename']].append(clss + bbox + cmmt)

    annotate = AnnotateWorker('/Users/dymello/usts/Annotations', '/Users/dymello/usts/raw', '/Users/dymello/usts/Images')
    p.map(annotate, enumerate(images_dict.items(), 0))

    print_flush('Done.')
    print_flush('In total %d images.' % len(images_dict))
    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))

    if not os.path.exists('/Users/dymello/usts/pickles'):
        os.mkdir('/Users/dymello/usts/pickles')
    pickle.dump(images_dict, open('/Users/dymello/usts/pickles/images_dict.pkl', 'wb'))

    ############################################################
    #  Split datasets
    ############################################################
    print_flush('Shuffling and splitting datasets', end=' ... ')
    t1 = time.time()

    if not os.path.exists('/Users/dymello/usts/ImageSets'):
        os.mkdir('/Users/dymello/usts/ImageSets')

    proportion = 0.8
    split_point = int(len(images_dict) * proportion)
    random.seed(0)
    clean_set_all = list(range(0, len(images_dict)))
    random.shuffle(clean_set_all)
    clean_set_trn = clean_set_all[:split_point]
    clean_set_tst = clean_set_all[split_point:]

    with open('/Users/dymello/usts/ImageSets/train_clean.txt', 'w') as f:
        f.write('\n'.join(['%07d' % x for x in clean_set_trn]))
    with open('/Users/dymello/usts/ImageSets/test_clean.txt', 'w') as f:
        f.write('\n'.join(['%07d' % x for x in clean_set_tst]))

    pickle.dump(clean_set_trn, open('/Users/dymello/usts/pickles/clean_set_trn.pkl', 'wb'))
    pickle.dump(clean_set_tst, open('/Users/dymello/usts/pickles/clean_set_tst.pkl', 'wb'))

    print_flush('Done.')
    print_flush('Clean dataset:')
    print_flush('    Training: %d clean' % len(clean_set_trn))
    print_flush('    Testing:  %d clean' % len(clean_set_tst))

    t2 = time.time()
    print_flush('Time elapsed: %f s.\n' % (t2 - t1))
