@echo off
echo Starting CARLA in headless mode (no rendering)...
cd "C:\Users\EcoCAR\CARLA_0.9.16\"
CarlaUE4.exe -dx11 -RenderOffScreen -quality-level=Low -fps=20
