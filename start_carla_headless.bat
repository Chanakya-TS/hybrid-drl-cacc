@echo off
echo Starting CARLA in headless mode (no rendering)...
cd /d "d:\Root\College\EcoCar\Research\CARLA_0.9.16"
CarlaUE4.exe -dx11 -RenderOffScreen -quality-level=Low -fps=20
