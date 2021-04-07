function [base, frames, img_size] = mpii_get_multiperson_test_seq_info(seq_id)

mpii_mupots_config

img_base_path = mpii_mupots_path;
base = fullfile(img_base_path, sprintf('TS%d/',seq_id));
frames = dir(sprintf('%s*.jpg',base));

img_size = [2048 2048];

if(seq_id>5) 
    img_size = [1080 1920];
end

end
