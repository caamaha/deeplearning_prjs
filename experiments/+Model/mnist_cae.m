function model = mnist_cae(model)
model.solver_cae1_def_file = fullfile(pwd, 'models', 'mnist_cae', 'solver_cae1.prototxt');
model.solver_cae1_classify_def_file = fullfile(pwd, 'models', 'mnist_cae', 'solver_cae1_classify.prototxt');
model.solver_cae2_def_file = fullfile(pwd, 'models', 'mnist_cae', 'solver_cae2.prototxt');

model.test_cae1_classify_def_file = fullfile(pwd, 'models', 'mnist_cae', 'test_cae1_classify.prototxt');