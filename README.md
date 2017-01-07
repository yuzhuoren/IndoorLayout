This code implements indoor layout estimation method published in: "A Coarse-to-Fine Indoor Layout Estimation (CFILE) Method" by Yuzhuo Ren, Shangwen Li, Chen Chen and C.-C. Jay Kuo, in ACCV 2016.

1.Download and prepare 'caffe-future' available at https://github.com/longjon/caffe/tree/future
2.Prepare data and run create_lmdb.py to generate layout boundary lmdb file and semantic surface lmdb file.
3.Run solve.py to train the multi-task FCN. The weight initialization can be downloaded here:https://gist.github.com/longjon/16db1e4ad3afc2614067. The trained model can be download here:https://www.dropbox.com/s/tyrbuafu7ejcq80/train_iter_454400.caffemodel?dl=0.
4.Run demo.m to test images. You can download the result of LSUN 2016 test here:https://www.dropbox.com/sh/r4z5oksewxwqsai/AADoYpn5VrG8nNd5mfZftjYda?dl=0.
The dataset we use is a processed version of LSUN 2016 dataset.The semantic surface of LSUN 2016 training and validation data are relabeled.

Label convention: 1-> Frontal wall, 2-> Left wall, 3-> Right wall, 4-> Floor, 5-> Ceiling

The preprocessed data can be download here:https://www.dropbox.com/s/85n95ftlp2rn0fq/LSUN2016_surface_relabel.zip?dl=0.

The multi-task FCN structure is first published in "Learning Informative Edge Maps for Indoor Scene Layout Prediction" by Arun Mallya and Svetlana Lazebnik in ICCV 2015. If you use the code, you are also required to cite this paper.

January, 2017
