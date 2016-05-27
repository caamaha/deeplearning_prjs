function active_caffe_mex(gpu_id)
% active_caffe_mex(gpu_id, caffe_version)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    % set gpu in matlab
    gpuDevice(gpu_id);

    cur_dir = pwd;
    %caffe_dir = fullfile(pwd, 'external', 'caffe');
    caffe_dir = '/home/soe/workspace/deep_learning/sw/caffe/matlab'
    
    if ~exist(caffe_dir, 'dir')
        warning('Specified caffe folder (%s) is not exist, change to default one (%s)', ...
            caffe_dir, fullfile(pwd, 'external', 'caffe', 'matlab'));
        caffe_dir = fullfile(pwd, 'external', 'caffe', 'matlab');
    end
    
    addpath(genpath(caffe_dir));
    cd(caffe_dir);
    caffe.set_device(gpu_id-1);
    cd(cur_dir);
end
