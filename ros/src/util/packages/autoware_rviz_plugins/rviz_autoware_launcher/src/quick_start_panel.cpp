#include "quick_start_panel.h"
#include "ui_quick_start_panel.h"
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

Quick_Start_Panel::Quick_Start_Panel(QWidget *parent) :
    rviz::Panel(parent),
    ui(new Ui::Quick_Start_Panel)
{
  ui->setupUi(this);
//  QPalette pal = palette();
//  pal.setColor(QPalette::Window, Qt::green);
//  this->setPalette(pal);
//  this->show();
//  this->setObjectName("sysinfogui");

//  ui->checkBox->setStyleSheet("color: red;"
//                              "background-color: yellow;");
}

Quick_Start_Panel::~Quick_Start_Panel()
{
  delete ui;
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(Quick_Start_Panel, rviz::Panel)
