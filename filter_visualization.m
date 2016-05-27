model                       = Model.natural_cae;
net = caffe.Net(model.test_cae1_def_file, fullfile('output', 'natural_cae', 'snapshot_iter_10000.caffemodel'), 'test');

e1_w = net.layers('encode1').params(1).get_data();

f1 = e1_w(:, :, :, 20);
f1 = f1*10 + 1;
f1 = f1 / 2;
imshow(f1);