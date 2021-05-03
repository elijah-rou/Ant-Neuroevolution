import sys, random
import pickle
from PySide2 import QtCore, QtGui, QtWidgets

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

        self.chro_input = QtWidgets.QLineEdit()
        self.chro_input.setPlaceholderText("Chromosome")

        self.file_button = QtWidgets.QPushButton('Select file')
        self.file_button.clicked.connect(self.select_file)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.start)

        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_layout.addWidget(self.chro_input)
        self.control_layout.addWidget(self.start_button)

        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(self.control_layout)

        self.dock = QtWidgets.QDockWidget("Controls", self)
        self.dock.setWidget(self.controls)

        # self.board.start()
        self.center()

    def update(self):
        super.update()
        self.statusbar.showMessage(self.board.environ.nest.food)
    
    def start(self):
        self.board.start(self.chro_input.text())

    def center(self):

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2
        )

    def select_file(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            file_path = dialog.selectedFiles()[0]
        pickle_off = open(file_path, "rb")
        emp = pickle.load(pickle_off)
        self.board.environ = Environment(emp)

class Communicate(QtCore.QObject):
    msgToSB = QtCore.Signal(str)


class Board(QtWidgets.QFrame):

    BoardWidth = 50
    BoardHeight = 50
    Timer = 200

    def __init__(self, parent):
        super(Board, self).__init__()

        # self.timer = QtCore.QBasicTimer()
        self.environ = Environment(h=Board.BoardHeight, w=Board.BoardWidth)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.c = Communicate()

    def start(self, chromosome=None):
        if not chromosome:
            self.environ.default_setup()
        else:
            self.environ.dominant_setup(chromosome)
        self.timer = QtCore.QBasicTimer()
        self.update()
        self.timer.start(Board.Timer, self)

    def squareWidth(self):
        return self.contentsRect().width() / Board.BoardWidth

    def squareHeight(self):
        return self.contentsRect().height() / Board.BoardHeight

    def paintEvent(self, event):

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
            row, col = agent.get_coord()
            self.drawSquare(
                painter,
                rect.left() + col * self.squareWidth(),
                boardTop + (Board.BoardHeight - row - 1) * self.squareHeight(),
            )
        painter.end()

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            self.environ.update()
            self.update()
            score = self.environ.nest.food
            self.c.msgToSB.emit("Food Collected: " + str(score))
        QtWidgets.QFrame.timerEvent(self, event)

    def drawSquare(self, painter, x, y, cell=None):

        colorTable = [
            0x000000,
            0xCC6666,
            0x66CC66,  # Green
            0x6666CC,
            0xCCCC66,  #
            0xCC66CC,  # purple
            0x66CCCC,  # Cyan
            0xDAAA00,  # Yellow
        ]
        if not cell:  # Pass in None if it is an Ant
            color = QtGui.QColor(0xCC0000)
        elif cell.is_nest:
            color = QtGui.QColor(0xDAAA00)
        elif not cell.active:
            color = QtGui.QColor(0xDAAA00)  # Draw Wall
        elif cell.pheromone > 0:  # draw pheromone
            color = QtGui.QColor.fromHsv(233, 255 * min(cell.pheromone, 1), 255)
        elif cell.food > 0:  # Draw Food
            color = QtGui.QColor(0x66CC66)
        elif cell.pheromone == 0:  # Draw blank space
            color = QtGui.QColor(0xFFFFFF)

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
