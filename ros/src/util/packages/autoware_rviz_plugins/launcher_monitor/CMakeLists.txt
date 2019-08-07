cmake_minimum_required(VERSION 2.8.3)
project(launcher_monitor)
add_compile_options(-std=c++11)

#find_package(autoware_build_flags PATHS /home/tu/git/Autoware) #delete in the future
find_package(catkin REQUIRED COMPONENTS rosconsole rviz )
find_package(Qt5 ${rviz_QT_VERSION} EXACT REQUIRED Core Widgets Network)
set(QT_LIBRARIES Qt5::Widgets Qt5::Network)

include_directories(${catkin_INCLUDE_DIRS})

## System dependencies are found with CMake's conventions
link_directories(${catkin_LIBRARY_DIRS})

##include(${QT_USE_FILE})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_definitions(-D_FILE_OFFSET_BITS=64)
add_definitions(-DQT_NO_KEYWORDS)

set(UI_FILES
  src/launcher_monitor.cpp
)

set(FORM_FILES
  src/launcher_monitor.ui
)

QT5_WRAP_CPP(QT_MOC_SRC_HPP
  src/launcher_monitor.h
)

QT5_WRAP_UI(QT_FORMS_HPP
  ${FORM_FILES}
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS rosconsole roscpp rviz topic_tools
  DEPENDS Boost)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(SYSTEM
  ${BOOST_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  include
)

## Declare a cpp library
add_library( ${PROJECT_NAME}
  ${QT_MOC_SRC_HPP}
  ${QT_FORMS_HPP}
  ${SOURCE_FILES}
  ${UI_FILES}
  ${QT_RESOURCES_CPP}
)

## Specify libraries to link a library or executable target against
target_link_libraries( ${PROJECT_NAME}
   ${QT_LIBRARIES}
   ${catkin_LIBRARIES}
   ${Boost_LIBRARIES}
)

#find_package (class_loader)
#class_loader_hide_library_symbols( ${PROJECT_NAME} )

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS
  ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  plugin_description.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY icons
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/icons
)
