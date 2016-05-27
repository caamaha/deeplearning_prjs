function made = mkdir_if_missing(path)
made = false;
if ~exist(fullfile(pwd, 'startup_.m'), 'file')
    error('Please change to the project root directory.');
end
    
if exist(path, 'dir') == 0
  mkdir(path);
  made = true;
end
