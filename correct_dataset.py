import os
import shutil
import glob
import numpy as np

error_dir = './error_files'
error_files = os.listdir(error_dir)
source_files = glob.glob('/data3/honglinc/tdw_datasets/playroom_mv_v1/*correct/*frame5_img0.png')
source_files = np.random.permutation(source_files)
filter_source_files = []

for source_file in source_files:
    source_scene_idx = source_file.split('_frame')[0]
    source_file_list = glob.glob(source_file.split('_frame')[0] + '*')
    if len(source_file_list) == 29:
        filter_source_files.append(source_file)
source_files = filter_source_files

correct_folder = './correct_files'
count = 0

breakpoint()
for error_file in error_files:

    error_scene_idx = error_file.split('_frame')[0]

    source_file = source_files[count]
    source_scene_idx = source_file.split('_frame')[0]
    source_file_list = glob.glob(source_file.split('_frame')[0]+'*')
    assert len(source_file_list) == 29, len(source_file_list)

    dst_file_list = [i.replace(source_scene_idx, error_scene_idx) for i in source_file_list]

    print('Error idx', error_scene_idx)

    for src in source_file_list:
        dst = src.replace(source_scene_idx, error_scene_idx)
        dst = os.path.join(correct_folder, dst)
        print('\t', '{0: <100}'.format(src), dst)
        shutil.copyfile(src, dst)

    count += 1
    print('-----')
print('Total corrected files: ', count, len(error_files))