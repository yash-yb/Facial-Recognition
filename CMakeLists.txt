cmake_minimum_required(VERSION 3.16)
project(FacialRecognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn objdetect videoio highgui)

add_library(facerec
  src/engine.cpp
  src/gallery.cpp
  src/camera.cpp
)

target_include_directories(facerec
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(facerec PUBLIC ${OpenCV_LIBS})

add_executable(facerec_cli examples/cli.cpp)
target_link_libraries(facerec_cli PRIVATE facerec)

add_executable(facerec_server server/main.cpp)
target_include_directories(facerec_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(facerec_server PRIVATE facerec)

if(WIN32)
  target_link_libraries(facerec_server PRIVATE ws2_32)
endif()

install(TARGETS facerec facerec_cli facerec_server
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(DIRECTORY include/facerec DESTINATION include)
