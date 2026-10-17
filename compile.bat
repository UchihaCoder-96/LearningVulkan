@echo off
C:/VulkanSDK/1.4.357.0/bin/slangc.exe shaders/shader.slang -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vertMain -entry fragMain -o shaders/slang.spv
copy /y shaders\slang.spv out\build\x64-debug\LearningVulkan