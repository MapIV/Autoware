#ifndef QUICK_START_PANEL_H
#define QUICK_START_PANEL_H

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
class Quick_Start_Panel;
}

class Quick_Start_Panel: public rviz::Panel
{
    Q_OBJECT

public:
  explicit Quick_Start_Panel(QWidget *parent = 0);
  ~Quick_Start_Panel();

protected:

protected Q_SLOTS:

private:
  Ui::Quick_Start_Panel *ui;

};

#endif // AUTOWARE_LAUNCHER_H
