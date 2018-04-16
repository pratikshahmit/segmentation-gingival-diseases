PATH_ROOT = '/Users/amanrana/Documents/perio_frame_validation/';
patients = strsplit(ls(PATH_ROOT));
patients(strcmp('', patients)) = [];

for patient = patients
    path = strcat(PATH_ROOT, patient, '/masks_roma/');
    disp(path);
    imgs = strsplit(ls(char(path)));
    imgs(strcmp('', imgs)) = [];
    
    for img = imgs
        path_img = strcat(path, img);
        disp(path_img);
        im = imread(char(path_img));
        [a, b] = trash_teeth(im);
        mask_ = strel('square', 5);
        a = imdilate(a, mask_);
        a = imgaussfilt(uint8(a), 1);
        a = a*255;
        dir_to_write = char(strcat(PATH_ROOT, patient, '/masks_roma_binary_images/'));
        mkdir(char(dir_to_write));
        path_to_write = char(strcat(dir_to_write, img));
        imwrite(a, path_to_write);
    end
end