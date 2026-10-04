add_executable(LearningVulkan
    LearningVulkan.cpp
    LearningVulkan.h
)

target_link_libraries(LearningVulkan PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
#    tinyobjloader
)

