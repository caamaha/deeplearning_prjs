function dataset = MPI_Sintel_complete(dataset)

cache_file = fullfile(pwd, 'output', 'cache', 'imdb.mat');

try
    load(cache_file);
catch
    imdb.images_path = {};
    imdb = scan_folders(imdb, fullfile('datasets', 'MPI-Sintel-complete', 'training', 'albedo'));
    imdb = scan_folders(imdb, fullfile('datasets', 'MPI-Sintel-complete', 'training', 'clean'));
    imdb = scan_folders(imdb, fullfile('datasets', 'MPI-Sintel-complete', 'training', 'final'));
    
    mkdir_if_missing(fullfile(pwd, 'output', 'cache'));
    save(cache_file, 'imdb', '-v7.3');
end

dataset.imdb = imdb;

end

function imdb = scan_folders(imdb, img_path)
% imdb = scan_folders(imdb, img_path)
%  Scan images from all sub directory under img_path and write their path
%  into imdb.

folders = dir(img_path);
for k = 3 : length(folders)
    if folders(k).isdir
        imdb = scan_images(imdb, fullfile(img_path, folders(k).name));
    end
end

end

function imdb = scan_images(imdb, img_path)
% imdb = scan_image(imdb, img_path)
%  Scan images from img_path and write their path into imdb.

imgs = dir(img_path);

for k = 3 : length(imgs)
    path = fullfile(img_path, imgs(k).name);
    imdb.images_path{end+1} = path;
end

end