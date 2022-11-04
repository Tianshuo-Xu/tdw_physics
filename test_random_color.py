
import random
import numpy as np

rejected_colors = []
def random_color_exclude_list(exclude_list=None, exclude_range=0.65, hsv_brightness=None):
    import colorsys
    if hsv_brightness is None:
        rgb = [random.random(), random.random(), random.random()]
    else:
        assert(hsv_brightness <= 1 and hsv_brightness>0), hsv_brightness
        rgb = [random.random(), random.random(), random.random()]
        scale = ((1-hsv_brightness) * random.random() + hsv_brightness)/np.max(rgb)
        rgb = [r*scale for r in rgb]
    if exclude_list is None or len(exclude_list) == 0:
        return rgb

    while True:
      bad_trial = False
      for exclude in exclude_list:

        assert len(exclude) == 3, exclude
        if np.linalg.norm(np.array([exclude[i] - rgb[i] for i in range(3)])) < exclude_range: #sany([np.linalg.norm(exclude[i] - rgb[i]) < exclude_range for i in range(3)]):
          bad_trial = True
          print("exclude", [int(r*255) for r in rgb], np.linalg.norm(np.array([exclude[i] - rgb[i] for i in range(3)])), exclude)
          rejected_colors.append(rgb)
          break
      if bad_trial:
        if hsv_brightness is None:
            rgb = [random.random(), random.random(), random.random()]
        else:
            rgb = [random.random(), random.random(), random.random()]
            scale = ((1-hsv_brightness) * random.random() + hsv_brightness)/np.max(rgb)
            rgb = [r*scale for r in rgb]

            #scale = ((1-hsv_brightness) * random.random() + hsv_brightness)/np.max(rgb)

            #hsv = (random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0)
            #rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])

        #  rgb = [random.random(), random.random(), random.random()]
      else:
          break
    return rgb

colors = []

for i in range(100):
    #color = random_color_exclude_list(exclude_list=[[1.0, 0, 0], [246/255, 234/255, 224/255], [1.0, 1.0, 0.0]], hsv_brightness=0.7)
    color = random_color_exclude_list(exclude_list=[[246/255, 234/255, 224/255]], hsv_brightness=0.7)
    colors.append(color)
    print("picked  color", i, color)

colors = np.tile(np.expand_dims(np.stack(colors, axis=0), axis=1), (1,10,1))
idx = np.argsort(colors[:,0,0] * 255 * 255 + colors[:,0,1] * 255 + colors[:,0,2] )
colors = colors[idx]

rejected_colors = np.tile(np.expand_dims(np.stack(rejected_colors, axis=0), axis=1), (1,10,1))
idx = np.argsort(rejected_colors[:,0,0] * 255 * 255 + rejected_colors[:,0,1] * 255 + rejected_colors[:,0,2] )
rejected_colors = rejected_colors[idx]

import ipdb; ipdb.set_trace()
import matplotlib.pyplot as plt

fig,(ax, ax2, ax3) = plt.subplots(1,3)
ax.imshow(colors)
ax2.imshow(rejected_colors)
ax3.imshow(np.array([[[246/255, 234/255, 224/255]]]))
plt.show()
