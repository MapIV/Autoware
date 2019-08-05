#ifndef LAUNCHER_MONITOR_H
#define LAUNCHER_MONITOR_H

#include <rviz/panel.h>
#include <QWidget>
#include <QCheckBox>
#include <ros/ros.h>
#include <std_msgs/String.h>

namespace Ui {
class Launcher_Monitor;
}

class Launcher_Monitor: public rviz::Panel
{
    Q_OBJECT

public:
  explicit Launcher_Monitor(QWidget *parent = 0);
  ~Launcher_Monitor();

protected:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;

protected Q_SLOTS:

private Q_SLOTS:


private:
  Ui::Launcher_Monitor *ui;

  void statusCallback(const std_msgs::String::ConstPtr& msg);


};

#endif // LAUNCHER_MONITOR_H
