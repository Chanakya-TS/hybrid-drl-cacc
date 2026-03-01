@echo off
echo Starting CARLA with DirectX 11...
echo Quality: Medium (options: Low, Medium, High, Epic)
cd /d "D:\Root\College\EcoCar\Research\CARLA_0.9.16\"
CarlaUE4.exe -dx11 -quality-level=Medium -fps=20
