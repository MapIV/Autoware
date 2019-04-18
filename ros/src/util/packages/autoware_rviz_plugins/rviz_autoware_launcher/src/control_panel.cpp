#include "control_panel.h"
#include "ui_control_panel.h"
#include <QTimer>
#include <QFileDialog>
#include <QSlider>
#include <QCheckBox>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <yaml-cpp/yaml.h>

#include <ros/ros.h>
#include <ros/common.h>

Control_Panel::Control_Panel(QWidget *parent) :
    rviz::Panel(parent),
    ui(new Ui::Control_Panel)
{
  ui->setupUi(this);
}

Control_Panel::~Control_Panel()
{
  delete ui;
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(Control_Panel, rviz::Panel)
