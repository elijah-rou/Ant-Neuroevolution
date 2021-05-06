import sys, random
import pickle
import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets
from unpickle_results import get_best
from ant_nn.environ.Environment import Environment

# TODO
# * Set the grid to fixed size, don't allow reshape
# * Only paint cells that need to be updated?
# * Separate drawing agent and drawing cell?


class AntGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(AntGUI, self).__init__()

        self.setGeometry(300, 300, 800, 800)
        self.setWindowTitle("AntsGUI")

        self.statusbar = self.statusBar()

        self.board = Board(self)
        self.board.c.msgToSB[str].connect(self.statusbar.showMessage)
        self.setCentralWidget(self.board)

        self.chrom_input = QtWidgets.QLineEdit()
        self.chrom_input.setText('C:/Users/evere/Documents/CornellTech/SwarmRobotics/Ant-Neuroevolution/fetchant_1.pkl')
        self.chrom_input.setPlaceholderText("Chromosome")

        self.file_button = QtWidgets.QPushButton('Select file')
        self.file_button.clicked.connect(self.select_file)

        self.epoch_label = QtWidgets.QLabel('Epoch:')
        self.epoch_input = QtWidgets.QLineEdit()
        self.epoch_input.setText('-1')

        self.score_label = QtWidgets.QLabel('nth best score:')
        self.score_input = QtWidgets.QLineEdit()
        self.score_input.setText('0')

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.start)

        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_layout.addWidget(self.chrom_input)
        self.control_layout.addWidget(self.file_button)
        self.control_layout.addWidget(self.epoch_label)
        self.control_layout.addWidget(self.epoch_input)
        self.control_layout.addWidget(self.score_label)
        self.control_layout.addWidget(self.score_input)
        self.control_layout.addWidget(self.start_button)

        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(self.control_layout)

        self.dock = QtWidgets.QDockWidget("Controls", self)
        self.dock.setFloating(False)
        self.dock.setWidget(self.controls)

        # self.board.start()
        self.center()

    def update(self):
        super.update()
        self.statusbar.showMessage(self.board.environ.nest.food)

    def start(self):
        chrom_file = self.chrom_input.text()
        if len(chrom_file) > 0:
            epoch_n = int(self.epoch_input.text())
            order_n = int(self.score_input.text())
            pickle_off = open(chrom_file, "rb")
            temp = pickle.load(pickle_off)
            chroms = np.array(temp[0][epoch_n])
            scores = np.array(temp[1][epoch_n]).argsort()
            chrom = chroms[scores[order_n]]

            chrom = get_best(temp)
            self.board.start(chrom)
        else:
            self.board.start()

    def center(self):

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2
        )

    def select_file(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        dialog.setViewMode(QtWidgets.QFileDialog.Detail)
        if dialog.exec_():
            self.chrom_input.setText(dialog.selectedFiles()[0])
        #     print(dialog.selectedFiles())
        #     file_path = dialog.selectedFiles()[0]
        # self.chrom_input.setText(file_path)

class Communicate(QtCore.QObject):
    msgToSB = QtCore.Signal(str)


class Board(QtWidgets.QFrame):

    BoardWidth = 30
    BoardHeight = 30
    Timer = 200

    def __init__(self, parent):
        super(Board, self).__init__()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.environ = None
        self.c = Communicate()

    def start(self, chromosome=None):
        self.environ = Environment(chromosome)
        # if not chromosome:
        #     self.environ.default_setup()
        # else:
        #     self.environ.dominant_setup(chromosome)
        self.timer = QtCore.QBasicTimer()
        self.update()
        self.timer.start(Board.Timer, self)

    def squareWidth(self):
        return self.contentsRect().width() / Board.BoardWidth

    def squareHeight(self):
        return self.contentsRect().height() / Board.BoardHeight

    def paintEvent(self, event):
        if not self.environ:
            return

        painter = QtGui.QPainter(self)
        rect = self.contentsRect()

        boardTop = rect.bottom() - Board.BoardHeight * self.squareHeight()

        for row in range(Board.BoardHeight):
            for col in range(Board.BoardWidth):
                cell = self.environ.grid[row][col]
                self.drawSquare(
                    painter,
                    rect.left() + col * self.squareWidth(),
                    boardTop + (Board.BoardHeight - row - 1) * self.squareHeight(),
                    cell,
                )
        for agent in self.environ.agents:
            has_food = agent.has_food
            row, col = agent.get_coord()
            self.drawSquare(
                painter,
                rect.left() + col * self.squareWidth(),
                boardTop + (Board.BoardHeight - row - 1) * self.squareHeight(),
                None,
                agent
            )
        painter.end()

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            self.environ.update()
            self.update()
            score = self.environ.nest.food
            self.c.msgToSB.emit("Food Collected: " + str(score))
        QtWidgets.QFrame.timerEvent(self, event)

    def drawSquare(self, painter, x, y, cell=None, ant=None):

        colorTable = [
            0x000000,
            0xCC6666,
            0x66CC66,  # Green
            0x6666CC,
            0xCCCC66,  #
            0xCC66CC,  # purple
            0x66CCCC,  # Cyan
            0xDAAA00,  # Yellow
            0xFFC0CB,  # Pink
            0xFFA500   # Orange
        ]
        if ant:  # Pass in None if it is an Ant
            if ant.has_food:
                color = QtGui.QColor(0xFFD700)
            else:
                color = QtGui.QColor(0xCC0000)
        elif cell.is_nest:
            color = QtGui.QColor(0xDAAA00)
        elif not cell.active:
            color = QtGui.QColor(0xDAAA00)  # Draw Wall
        elif cell.pheromone > 0:  # draw pheromone
            color = QtGui.QColor.fromHsv(233, 255 * min(cell.pheromone, 1), 255)
        elif cell.food > 0:  
            color = QtGui.QColor(0x66CC66)# Draw Food
        elif cell.pheromone == 0: 
            color = QtGui.QColor(0xFFFFFF) # Draw blank space
        elif cell.pheromone <0:
            color = QtGui.QColor(0xFFC0CB)

        painter.fillRect(
            x + 1, y + 1, self.squareWidth() - 1, self.squareHeight() - 1, color
        )


class Things(object):

    Empty = 0
    Phero = 1
    Wall = 2
    Nest = 3
    Food = 4
    AgentNoFood = 5
    AgentFood = 6
    NoShape = 0
    # ZShape = 1
    # SShape = 2
    # LineShape = 3
    # TShape = 4
    # SquareShape = 5
    # LShape = 6
    # MirroredLShape = 7


def main():

    app = QtWidgets.QApplication(sys.argv)
    t = AntGUI()
    t.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
