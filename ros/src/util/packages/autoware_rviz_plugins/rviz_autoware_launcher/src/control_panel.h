#ifndef CONTROL_PANEL_H
#define CONTROL_PANEL_H

#include <QMainWindow>
#include <QWidget>
#include <QEvent>
#include <QKeyEvent>
#include <QStatusBar>
#include <QImage>
#include <QPainter>
#include <QLabel>
#include <QTabWidget>
#include <QTime>

#include "rviz/visualization_manager.h"
#include "rviz/render_panel.h"
#include "rviz/display.h"
#include "rviz/panel.h"
#include "rviz/default_plugin/tools/measure_tool.h"
#include "rviz/tool_manager.h"
#include "rviz/default_plugin/tools/point_tool.h"

namespace Ui {
class Control_Panel;
}

class Control_Panel : public rviz::Panel
{
    Q_OBJECT

public:
  explicit Control_Panel(QWidget *parent = 0);
  ~Control_Panel();

protected:

protected Q_SLOTS:

private:
  Ui::Control_Panel *ui;

};

#endif // AUTOWARE_LAUNCHER_H
