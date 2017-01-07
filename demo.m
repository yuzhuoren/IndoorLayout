clear all;
close all;
clc;

image_dir = '';
result_dir = '';

if ~exist(result_dir,'dir')
    mkdir(result_dir);
    
end
mkdir([result_dir,'Edge/']);
mkdir([result_dir,'GC/']);
% change this as required (see README.txt)
CAFFE_ROOT = '';

% model info
MODEL_FILE    = 'prototxts/deploy.prototxt';
SNAPSHOT_FILE = 'model/train.caffemodel';
mean_val.R = 0.515988;
mean_val.G = 0.456014;
mean_val.B = 0.400506;

% initialize caffe
addpath(genpath(CAFFE_ROOT))
addpath('..');
use_gpu = 1;
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);
caffe.reset_all();
%  if caffe('is_initialized')
% caffe('reset')
%  end
net = caffe.Net(MODEL_FILE, SNAPSHOT_FILE, 'test');
% matcaffe_init(use_gpu, MODEL_FILE, SNAPSHOT_FILE);
% the image can be passed as is, without rescaling, but we use a size
% of 404X404 in the deploy network. The FCN can run on arbitrary sized
% inputs
im_ht = 404; im_wd = 404;

files = dir(fullfile(image_dir,'*.jpg'));
for i = 1 : length(files)
    
    
    % load and process image
    filename = files(i).name;
    [im, original_im] = prepare_image([image_dir,filename], im_ht, im_wd, mean_val);
    [h, w, ~] = size(original_im);
    
    % forward pass
    %scores = caffe('forward', {im});
    scores = net.forward({im});
    
    % get BB
    BB = scores{1};
    BB = permute(BB, [2, 1, 3]);
    BB = imresize(BB, [h, w]);
    % BB has two classes: not-edge/edge. The sum of the values across the
    % 3rd dimension is 1 for each pixel
    edge_prob_map = BB(:, :, 2);
    
    imwrite(edge_prob_map,[result_dir,'Edge/',filename]);
    
    
    % figure
    % subplot(1, 2, 1), imshow(original_im/255), title('Input Image')
    % subplot(1, 2, 2), imshow(edge_prob_map, []), ...
    %     title('Informative Edge Probability Map')
    % fprintf(['This is an image from the Hedau test set.' ...
    %     '\nNote that the uninformative edges on the walls due to the ' ...
    %     'wood texture, as well as the window and shelves are ignored!\n']);
    
    % get GC
    % these predictions are very noisy due to lack of a CRF
    GC = scores{2};
    GC = permute(GC, [2, 1, 3]);
    GC = imresize(GC, [h, w], 'nearest');
    [~, GC_map] = max(GC, [], 3);
    
    % uncomment below to display GC map
    %figure, imshow(GC_map, []), colormap jet, title('Geometric Context Map')
    imwrite(GC_map./6,[result_dir,'GC/',filename(1:end-4),'.png']);
    
end


