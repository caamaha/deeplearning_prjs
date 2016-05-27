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


%% -------------------- Train ---------------------
caffe_solver = caffe.Solver(model.solver_cae1_def_file);
caffe.set_mode_gpu();
iter_ = caffe_solver.iter();
max_iter = caffe_solver.max_iter();

batch_size = 10;

while (iter_ < max_iter)
    caffe_solver.net.set_phase('train');
    start_index = randi(length(train_x) - batch_size + 1);
    im = single(train_x(start_index : start_index + batch_size - 1, :)) / 255;
    im = reshape(im, [], 28, 28, 1);
    im = permute(im, [2, 3, 4, 1]);
    
    net_inputs = {im};
    
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    rst = caffe_solver.net.get_output();
    
    iter_ = caffe_solver.iter();
end

%caffe.reset_all();

% save final model
model_path = fullfile('output', 'mnist_cae', 'stage1_model');
caffe_solver.net.save(model_path);

% Visualize
% restruction = caffe_solver.net.blobs('flatdecode').get_data();
% restruction = reshape(restruction, 28, 28, []);
% subplot(121);imshow(im);
% subplot(122);imshow(restruction);


%end