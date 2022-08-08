# General settings for all rooms
commands.extend([
            add_scene,
            {"$type": "set_post_exposure",
             "post_exposure": 0.4},
            {"$type": "set_ambient_occlusion_intensity",
             "intensity": 0.175},
            {"$type": "set_ambient_occlusion_thickness_modifier",
             "thickness": 3.5},
            {"$type": "set_aperture", "aperture": 1.5},
        ])

# Specific settings for different rooms
if self.room == 'tdw_room':
    commands.extend([
        {"$type": "adjust_directional_light_intensity_by", "intensity": 0.25},
        {"$type": "adjust_point_lights_intensity_by", "intensity": 0.6},
        {"$type": "set_shadow_strength", "strength": 0.5},
        {"$type": "rotate_directional_light_by", "angle": -30, "axis": "pitch", "index": 0},
    ])
elif self.room == 'mm_craftroom_1b':
    commands.extend([
        {"$type": "adjust_point_lights_intensity_by", "intensity": 0.6},
        {"$type": "adjust_directional_light_intensity_by", "intensity": 0.6},
        {"$type": "set_shadow_strength", "strength": 0.5}])
elif self.room == 'box':

    commands.extend([
        {"$type": "adjust_directional_light_intensity_by", "intensity": 0.7},
        {"$type": "set_shadow_strength", "strength": 0.6},
    ])
elif self.room == 'archviz_house':
    commands.extend([
     {"$type": "adjust_directional_light_intensity_by", "intensity": 0.4},
     {"$type": "adjust_point_lights_intensity_by", "intensity": 0.5},
     {"$type": "set_shadow_strength", "strength": 0.8}])
else:
    pass