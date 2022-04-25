
import h5py
import glob
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np


trial = '/home/honglinc/tdw_physics/rotate_data/model_split_0/0000.hdf5'
passes = ['_img', '_id', '_normal']
num_views = 4
f = h5py.File(trial, "r")


# for fid, frame_num in enumerate(list(f["frames"].keys())[1:]):
#     print(fid, frame_num)
#     data = f["frames"][frame_num]['images'] # data for each frame
#     fig, axs = plt.subplots(1, len(passes), figsize=(30, 10))
#     for i, _pass in enumerate(passes):
#         for k, v in data.items():
#             if _pass in k:
#                 cam_id = int(k.split('cam')[-1])
#
#                 if cam_id == (fid + 1):
#                     img = Image.open(io.BytesIO(data[k][:]))
#                     print(data[k])
#                     axs[i].imshow(img)
#                     axs[i].set_axis_off()
#     plt.savefig('./tmp/%s.png' % frame_num, bbox_inches='tight')
#     plt.show()
#     plt.close()

for frame_num in list(f["frames"].keys())[1:]:
    print(frame_num)
    data = f["frames"][frame_num]['images'] # data for each frame
    fig, axs = plt.subplots(num_views, len(passes), figsize=(8, 10))
    for i, _pass in enumerate(passes):
        for k, v in data.items():
            if _pass in k:
                cam_id = int(k.split('cam')[-1])
                try:
                    img = Image.open(io.BytesIO(data[k][:]))
                    print(data[k])
                except:
                    breakpoint()

                if False: #_pass == '_flow':
                    breakpoint()
                    axs[cam_id, i].imshow(np.array(Image) * 10)
                else:
                    axs[cam_id, i].imshow(img)
                axs[cam_id, i].set_xticks([])
                axs[cam_id, i].set_yticks([])
                if i == 0:
                    axs[cam_id, i].set_ylabel('Camera %d' % cam_id, fontsize=19)
                if cam_id == 0:
                    axs[cam_id, i].set_title(_pass[1:], fontsize=19)
    plt.savefig('./tmp/%s.png' % frame_num)
    plt.show()
    plt.close()


# Create video
def create_video(frames, video_name, save_path):
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()

