import pybullet as p
import pybullet_data as pd
import os


filename = "/home/htung/Documents/2021/tdw_physics/log2/0000_obj1.obj"
p.connect(p.DIRECT)
name_in = filename
name_out = "/home/htung/Documents/2021/tdw_physics/log2/0000_vhacd1.obj"
name_log = "log.txt"
p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000)