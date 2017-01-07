function [im, original_im] = prepare_image(filename, im_ht, im_wd, mean_val)

    % load image and resize to network input size
    im = single(imread(filename));
    original_im = im;
    im = imresize(im, [im_ht, im_wd], 'bilinear');
    % subtract mean and normalize
    im(:, :, 1) = im(:, :, 1) - mean_val.R * 255;
    im(:, :, 2) = im(:, :, 2) - mean_val.G * 255;
    im(:, :, 3) = im(:, :, 3) - mean_val.B * 255;
    % permute RGB to BGR and flip width and height
    im = im(:, :, [3, 2, 1]);
    im = permute(im, [2, 1, 3]);
    
end