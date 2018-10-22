cmake_minimum_required(VERSION 2.8.12)
project(trainers)
SET(dlib_DIR $ENV{HOME}/src/dlib CACHE PATH "Dlib source path")

add_subdirectory(${dlib_DIR} dlib_build)


add_executable(dlib_face_detection_trainer dlib_face_detection_trainer.cpp)
target_link_libraries(dlib_face_detection_trainer dlib::dlib)

add_executable(dlib_face_landmarks_trainer dlib_face_landmarks_trainer.cpp)
target_link_libraries(dlib_face_landmarks_trainer dlib::dlib)

add_executable(csv_index_builder csv_index_builder.cpp)
target_link_libraries(csv_index_builder)