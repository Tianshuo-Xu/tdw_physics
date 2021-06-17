set i=0
for %%a in (0 1 2) do (
   set /A i+=1
   set "seeds[!i!]=%%a"
)
set i=0
for %%a in (15 100 1000) do (
   set /A i+=1
   set nums[!i!]=%%a
)
set i=0
for %%a in (test val train) do (
   set /A i+=1
   set groups[!i!]=%%a
)
FOR /L %%i in (1,1,%i%) do (
    FOR /D %%d in (c:/Users/hsiaoyut/human-physics-benchmarking/stimuli/generation/pilot-dominoes/*) DO ( \
        START C:/Users/hsiaoyut/.conda/envs/tdw/python.exe c:/Users/hsiaoyut/tdw_physics/tdw_physics/target_controllers/rolling_sliding.py \
        @c:/Users/eliwang/human-physics-benchmarking/stimuli/generation/pilot-dominoes/%%~nd/commandline_args.txt --height 256 --width 256 --training_data_mode --seed !seeds[%%i]! --num_multiplier !nums[%%i]! --save_passes "" --write_passes "_img,_id" --save_meshes
        TIMEOUT /T 60
    )
)



