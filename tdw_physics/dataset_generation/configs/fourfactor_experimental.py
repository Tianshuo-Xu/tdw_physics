set_a = {
    "container": "b04_bowl_smooth",
    "target": "cube",
    "distractor": "sphere"
}

set_b = {
    "container": "serving_bowl",
    "target": "pyramid",
    "distractor": "cylinder"
}

set_a_params = {
    "container_scale_range": "5.0",
    "container_mass_range": "3.0",
    "target_scale_range": "[[0.25,0.25],[1.0,1.0],[0.25,0.25]]",
    "target_angle_range": "0",
    "target_position_jitter": 0.0,
    "target_rotation_jitter": 0.0,
    "target_mass_range": "1.5",            
    "distractor_scale_range": "0.5",
    "distractor_position_range": "[1.0,1.5]",
    "distractor_rotation_range": "[0,0]",
    "distractor_rotation_jitter": 0.0,
    "distractor_mass_range": "2.0",                
    "force_scale_range": "15.0",
    "force_wait": 60,
}
set_a_params.update(set_a)

set_b_params = {
    "container_scale_range": "3.0",
    "container_mass_range": "3.0",    
    "target_scale_range": "[[0.5,0.5],[1.0,1.0],[0.5,0.5]]",
    "target_angle_range": "0",    
    "target_position_jitter": 0.0,
    "target_rotation_jitter": 0.0,
    "target_mass_range": "1.5",        
    "distractor_scale_range": "0.5",
    "distractor_position_range": "[1.0,1.5]",
    "distractor_rotation_range": "[0,0]",
    "distractor_rotation_jitter": 0.0,
    "distractor_mass_range": "1.5",
    "distractor_always_horizontal": True,
    "force_scale_range": "15.0",    
    "force_wait": 60
}
set_b_params.update(set_b)

contain_params = {
    "target_position_range": "0.0"
}

occlude_params = {
    "target_position_range": "0.6"
}

collide_params = {
    "force_angle_range": "0",
    "distractor_angle_range": "45"
}

miss_params = {
    "force_angle_range": "45",
    "distractor_angle_range": "45"
}

common_params = {
    "container_position_range": "[[-0.1,0.1],0.0,[-0.1,0.1]]",
    "target_always_vertical": True,
    "container_flippable": False,
    "scale_objects_uniformly": False,
    "match_probe_and_target_color": False,
    "target_angle_reflections": False,
    "push_distractor": True,
    "target_material": "parquet_wood_red_cedar",
    "container_material": "parquet_wood_red_cedar",
    "min_frames": 150,
    "max_frames": 300,
    "camera_min_height": 0.5,
    "camera_max_height": 1.5,
    "camera_radius": [2.0,3.0],
    "camera_left_right_reflections": False,
    "camera_min_angle": 0,
    "camera_max_angle": 0
}
