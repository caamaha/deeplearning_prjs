function startup()
% startup()
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    curdir = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(curdir, 'utils')));
    addpath(genpath(fullfile(curdir, 'functions')));
    addpath(genpath(fullfile(curdir, 'bin')));
    addpath(genpath(fullfile(curdir, 'experiments')));

    mkdir_if_missing(fullfile(curdir, 'datasets'));

    mkdir_if_missing(fullfile(curdir, 'external'));

    %caffe_path = fullfile(curdir, 'external', 'caffe', 'matlab');
    caffe_path = '/home/soe/workspace/deep_learning/sw/caffe/matlab';
    if exist(caffe_path, 'dir') == 0
        error('matcaffe is missing from external/caffe/matlab; See README.md');
    end
    addpath(genpath(caffe_path));

    mkdir_if_missing(fullfile(curdir, 'output'));
    
    mkdir_if_missing(fullfile(curdir, 'models'));

    fprintf('fast_flow startup done\n');
end
