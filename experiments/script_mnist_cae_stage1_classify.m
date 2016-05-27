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
caffe_solver = caffe.Solver(model.solver_cae1_classify_def_file);
caffe_solver.net.copy_from(fullfile('output', 'mnist_cae', 'stage1_model'));
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
    
    [~, labels] = max(train_y(start_index : start_index + batch_size - 1, :), [], 2);
    labels = reshape(labels, 1, 1, 1, 10) - 1;
    net_inputs = {im, labels};
    
    caffe_solver.net.reshape_as_input(net_inputs);

    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    rst = caffe_solver.net.get_output();
    
    iter_ = caffe_solver.iter();
end

% save final model
model_path = fullfile('output', 'mnist_cae', 'stage1_classify_model');
caffe_solver.net.save(model_path);

% test 


%end