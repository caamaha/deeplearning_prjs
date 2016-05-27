function model = natural_cae(model)
model.solver_cae1_def_file = fullfile(pwd, 'models', 'natural_cae', 'solver_cae1.prototxt');
model.test_cae1_def_file = fullfile(pwd, 'models', 'natural_cae', 'test_cae1.prototxt');
