import os
import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QApplication, QStackedWidget, QWidget

from commandexec import CommandExecutor, NoConnectionError, commandexec_stop, execute_command, COMMAND_SIT, COMMAND_STAND
from gestrec import Gestrec, gestrec_stop, gestrec_on, gestrec_off
import time
# Variables
ui_startwindow = 'ui/start_window.ui'
ui_control_window = 'ui/control_window.ui'

# Objects
commandexec = CommandExecutor()
gestrec = Gestrec()

# Class 1 of 2
class StartScreen(QDialog):
    # initialize startscreen (connect and exit, not in gesture recognition guy yet)
    def __init__(self):
        # super: parent class
        super(StartScreen, self).__init__()
        # load the pre-built UI (use PyQt5 library)
        loadUi(ui_startwindow, self)                            # Plug-in (platform-independent abstractions for GUIs)
        # assign handlers to buttons
        self.connectButton.clicked.connect(self.go_to_control)
        self.exitButton.clicked.connect(self.close_app)

    # is called, if "Connect to Robot" is clicked
    def go_to_control(self):
        try:
            # start process in parallel
            commandexec.start()             # return-value is ignored
            # instanciate ControlScreen-Object
            control = ControlScreen()
            # ControlScreen() inherits from QDialog -> is a widget
            widget.addWidget(control)
            # set control as current widget
            widget.setCurrentWidget(control)
        except NoConnectionError as ex:
            self.errorLabel.setText(ex.message)

    # is called, if "Exit" is clicked
    def close_app(self):
        commandexec_stop()
        sys.exit()

# Class 2 of 2
# is called from, if "Connect to Robot" is clicked
class ControlScreen(QDialog):       # inherits from QDialog
    def __init__(self):
        # super: parent class
        super(ControlScreen, self).__init__()
        # load the pre-built UI (use PyQt5 library)
        loadUi(ui_control_window, self)
        # assign handlers to buttons
        self.activateButton.clicked.connect(self.start_recognition)
        self.stopButton.clicked.connect(self.stop_recognition)
        self.exitButton.clicked.connect(self.close_app)
        self.sitButton.clicked.connect(self.sit_down)
        self.standButton.clicked.connect(self.stand_up)
        gestrec.start()

    # handlers of buttons
    def start_recognition(self):
        gestrec_on()
        self.recognitionLabel.setText("Gesture Recognition: on ")

    def stop_recognition(self):
        gestrec_off()
        self.recognitionLabel.setText("Gesture Recognition: off")

    def sit_down(self):
        execute_command(COMMAND_SIT)
    
    def stand_up(self):
        execute_command(COMMAND_STAND)

    def close_app(self):
        gestrec_stop()
        commandexec_stop()
        sys.exit()


# Main function (when running GUI)
if __name__ == '__main__':
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # sys.argv is a list. First element [0] is for example: ['c:/Users/yourUserName/git/zhaw_ba_robodog/app.py']
    app = QApplication(sys.argv)
    start = StartScreen()           # startscreen then calls controlscreen
    # start = ControlScreen()
    widget = QStackedWidget()
    widget.addWidget(start)
    widget.setFixedHeight(604)
    widget.setFixedWidth(701)
    widget.show()

    try:
        sys.exit(app.exec())
    except:
        print("Exiting")
