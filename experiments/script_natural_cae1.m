function script_natural_cae1()
% Natural convolutional autoencoder training and testing
% -------------------------------------------
% Copyright (c) 2016, Soe
% -------------------------------------------

clc;
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup_'));

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


%% -------------------- Train ---------------------
caffe_solver = caffe.Solver(model.solver_cae1_def_file);
caffe.set_mode_gpu();
iter_ = caffe_solver.iter();
max_iter = caffe_solver.max_iter();

train1_blob = zeros(64, 64, 3, 1000);
train2_blob = zeros(64, 64, 3, 1000);

for k = 1:100
    [train1_blob(:, :, :, (k-1)*10+1:k*10), train2_blob(:, :, :, (k-1)*10+1:k*10)] = generate_batch(dataset.imdb, 10, 64);
end


while (iter_ < max_iter)
    caffe_solver.net.set_phase('train');
    
    ind = randi(1000);
    net_inputs = {train2_blob(:, :, :, ind), train1_blob(:, :, :, ind)};
    
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    
    iter_ = caffe_solver.iter();
    
    % visualize
    if mod(iter_, 20) == 0
        im_i = caffe_solver.net.blobs('noisy_data').get_data();
        im_o = caffe_solver.net.blobs('decode1').get_data();
        subplot(211);imshow(im_i(:, :, :, 1));
        subplot(212);imshow(im_o(:, :, :, 1));
        pause(0.01);
    end
end


end

function [input1_blob, input2_blob] = generate_batch(imdb, batch_size, input_size)
im_data = single(imread(imdb.images_path{randi(length(imdb.images_path))})) / 255;
im_noisy = im_data .* single(rand(size(im_data)) > 0.2);

im_data = im_data(:, :, [3, 2, 1]);         % convert from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);      % permute width and height

im_noisy = im_noisy(:, :, [3, 2, 1]);         % convert from RGB to BGR
im_noisy = permute(im_noisy, [2, 1, 3]);      % permute width and height

[w, h, ~] = size(im_data);
input1_blob = zeros(input_size, input_size, 3, batch_size);
input2_blob = input1_blob;
parfor k = 1:batch_size
    ws = randi(w-input_size+1);
    hs = randi(h-input_size+1);
    input1_blob(:, :, :, k) = im_data(ws:ws+input_size-1, hs:hs+input_size-1, :);
    input2_blob(:, :, :, k) = im_noisy(ws:ws+input_size-1, hs:hs+input_size-1, :);
end

end