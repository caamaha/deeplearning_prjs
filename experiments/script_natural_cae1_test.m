function script_natural_cae1_test()
% Natural convolutional autoencoder training and testing
% -------------------------------------------
% Copyright (c) 2016, Soe
% -------------------------------------------

clc;
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

%% -------------------- CONFIG --------------------
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id);

% model
model                       = Model.natural_cae;
% cache base
cache_base                  = 'natural_cae';
% train/test data
dataset                     = [];
dataset                     = Dataset.MPI_Sintel_complete(dataset);


%% -------------------- Test ---------------------
net = caffe.Net(model.test_cae1_def_file, fullfile('output', 'natural_cae', 'snapshot_iter_10000.caffemodel'), 'test');
caffe.set_mode_gpu();


test_blob = zeros(64, 64, 3, 1000);

for k = 1:100
    test_blob(:, :, :, (k-1)*10+1:k*10) = generate_batch(dataset.imdb, 10, 64);
end

for k = 1:size(test_blob, 4)
    net_inputs = {test_blob(:, :, :, k)};

    net.reshape_as_input(net_inputs);
    net.set_input_data(net_inputs);
    net.forward_prefilled();
    
    % visualize
    im_i = net.blobs('data').get_data();
    im_o = net.blobs('decode1').get_data();
    subplot(211);imshow(im_i(:, :, :, 1));
    subplot(212);imshow(im_o(:, :, :, 1));
    pause(0.01);
    
end

end

function input_blob = generate_batch(imdb, batch_size, input_size)
im_data = imread(imdb.images_path{randi(length(imdb.images_path))});
im_data = im_data(:, :, [3, 2, 1]);         % convert from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);      % permute width and height
im_data = single(im_data) / 255;            % convert to single precision

[w, h, ~] = size(im_data);
input_blob = zeros(input_size, input_size, 3, batch_size);
parfor k = 1:batch_size
    ws = randi(w-input_size+1);
    hs = randi(h-input_size+1);
    input_blob(:, :, :, k) = im_data(ws:ws+input_size-1, hs:hs+input_size-1, :);
end

end