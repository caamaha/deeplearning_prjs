%function script_fast_flow_autoencoder()
% script_fast_flow_autoencoder()
% Fast flow autoencoder training and testing
% -------------------------------------------
% Copyright (c) 2016, Soe
% -------------------------------------------

clear;clc;
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));

%% -------------------- CONFIG --------------------
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id);

% model
model                       = Model.mnist_cae;
% cache base
cache_base                  = 'mnist_cae';
% train/test data
load('datasets/mnist_uint8.mat')


%% -------------------- Test ---------------------
net = caffe.Net(model.test_cae1_classify_def_file, fullfile('output', 'mnist_cae', 'stage1_classify_model'), 'test');

caffe.set_mode_gpu();


batch_size = 10;

start_index = randi(length(train_x) - batch_size + 1);
im = single(train_x(start_index : start_index + batch_size - 1, :)) / 255;
im = reshape(im, [], 28, 28, 1);
im = permute(im, [2, 3, 4, 1]);

[~, labels] = max(train_y(start_index : start_index + batch_size - 1, :), [], 2);
labels = reshape(labels, 1, 1, 1, 10) - 1;

net_inputs = {im};

net.reshape_as_input(net_inputs);
net.set_input_data(net_inputs);
net.forward_prefilled();
prob = net.blobs('prob').get_data();

%end