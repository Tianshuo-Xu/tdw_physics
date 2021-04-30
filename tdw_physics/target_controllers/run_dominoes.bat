setlocal EnableDelayedExpansion
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
for %%a in (val test train) do (
   set /A i+=1
   set groups[!i!]=%%a
)

FOR /L %%i in (1,1,3) do (        
    FOR /D %%d in (c:/Users/eliwang/human-physics-benchmarking/stimuli/generation/pilot-dominoes/*) DO (
        C:/Users/eliwang/.conda/envs/tdw/python.exe c:/Users/eliwang/tdw_physics/tdw_physics/target_controllers/dominoes.py @c:/Users/eliwang/human-physics-benchmarking/stimuli/generation/pilot-dominoes/%%~nd/commandline_args.txt --training_data_mode --dir "Z:/eliwang/dominoes/%%~nd/!groups[%%i]!" --port 1071 --save_passes "" --height 256 --width 256 --monochrome 0 --random 0 --seed !seeds[%%i]! --num !nums[%%i]!
    )
)

REM C:/Users/eliwang/.conda/envs/tdw/python.exe c:/Users/eliwang/tdw_physics/tdw_physics/target_controllers/dominoes.py @c:/Users/eliwang/human-physics-benchmarking/stimuli/generation/pilot-dominoes/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom/commandline_args.txt --training_data_mode --dir "Z:/eliwang/dominoes/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom" --port 1071 --save_passes "" --height 256 --width 256 --monochrome 0 --random 0 --seed 0 --num 15 
