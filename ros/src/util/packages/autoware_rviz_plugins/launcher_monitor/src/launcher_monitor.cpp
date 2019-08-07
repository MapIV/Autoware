#include "launcher_monitor.h"
#include "ui_launcher_monitor.h"

#include <QCheckBox>

#include <ros/console.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>


void Launcher_Monitor::statusCallback(const std_msgs::String::ConstPtr& msg)
{
  std::stringstream ss(msg->data.c_str());
  std::string item;
  std::vector<std::string> split_status;

  while (std::getline(ss, item, ' '))
  {
    if ((item[0] != 'r') && (item.length() > 1))
    {
      split_status.push_back(item.substr(0, 1));
      split_status.push_back(item.substr(2));
    }
    else
      split_status.push_back(item);
  }

  std::vector<std::string>::iterator ite = split_status.begin();
  while ((ite + 1) != split_status.end())
  {
    if ((*ite) == "root/planning/decision")
    {
      if (*(ite + 1) == "1"){
        ui->decision_status->setText(QString::fromStdString("On"));
        ui->decision_status->setStyleSheet("color: red;");}
      else{
        ui->decision_status->setText(QString::fromStdString("Off"));
        ui->decision_status->setStyleSheet("color: green;");}}
    if ((*ite) == "root/localization")
    {
      if (*(ite + 1) == "1"){
        ui->localization_status->setText(QString::fromStdString("On"));
        ui->localization_status->setStyleSheet("color: red;");}
      else{
        ui->localization_status->setText(QString::fromStdString("Off"));
        ui->localization_status->setStyleSheet("color: green;");}}
    if ((*ite) == "root/planning/mission")
    {
      if (*(ite + 1) == "1"){
        ui->mission_status->setText(QString::fromStdString("On"));
        ui->mission_status->setStyleSheet("color: red;");}
      else{
        ui->mission_status->setText(QString::fromStdString("Off"));
        ui->mission_status->setStyleSheet("color: green;");}}
    if ((*ite) == "root/planning/motion")
    {
      if (*(ite + 1) == "1"){
        ui->motion_status->setText(QString::fromStdString("On"));
        ui->motion_status->setStyleSheet("color: red;");}
      else{
        ui->motion_status->setText(QString::fromStdString("Off"));
        ui->motion_status->setStyleSheet("color: green;");}}
    if ((*ite) == "root/planning/prediction")
    {
      if (*(ite + 1) == "1"){
        ui->prediction_status->setText(QString::fromStdString("On"));
        ui->prediction_status->setStyleSheet("color: red;");}
      else{
        ui->prediction_status->setText(QString::fromStdString("Off"));
        ui->prediction_status->setStyleSheet("color: green;");}}
    if ((*ite) == "root/perception/object/tracking")
    {
      if (*(ite + 1) == "1"){
        ui->tracking_status->setText(QString::fromStdString("On"));
        ui->tracking_status->setStyleSheet("color: red;");}
      else{
        ui->tracking_status->setText(QString::fromStdString("Off"));
        ui->tracking_status->setStyleSheet("color: green;");}}

    ite += 1;
  }
}

Launcher_Monitor::Launcher_Monitor(QWidget *parent) :
    rviz::Panel(parent),
    ui(new Ui::Launcher_Monitor)
{
  ui->setupUi(this);

  ui->decision_status->setText(QString::fromStdString("Initialize"));
  ui->decision_status->setStyleSheet("color: yellow;");
  ui->localization_status->setText(QString::fromStdString("Initialize"));
  ui->localization_status->setStyleSheet("color: yellow;");
  ui->mission_status->setText(QString::fromStdString("Initialize"));
  ui->mission_status->setStyleSheet("color: yellow;");
  ui->motion_status->setText(QString::fromStdString("Initialize"));
  ui->motion_status->setStyleSheet("color: yellow;");
  ui->prediction_status->setText(QString::fromStdString("Initialize"));
  ui->prediction_status->setStyleSheet("color: yellow;");
  ui->tracking_status->setText(QString::fromStdString("Initialize"));
  ui->tracking_status->setStyleSheet("color: yellow;");

  sub_ = nh_.subscribe("/autoware_launcher/status", 500, &Launcher_Monitor::statusCallback, this);

}

Launcher_Monitor::~Launcher_Monitor()
{
  delete ui;
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(Launcher_Monitor, rviz::Panel)
