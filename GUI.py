import sys, random
from PySide2 import QtCore, QtGui, QtWidgets

from ant_nn.environ.Environment import Environment

# TODO
# * Set the grid to fixed size, don't allow reshape
# * Only paint cells that need to be updated?
# * Separate drawing agent and drawing cell?


class AntGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(AntGUI, self).__init__()

        self.setGeometry(300, 300, 1000, 1000)
        self.setWindowTitle("AntsGUI")
        self.board = Board(self)

        self.setCentralWidget(self.board)

        self.statusbar = self.statusBar()

        self.statusbar = self.statusBar()
        self.board.c.msgToSB[str].connect(self.statusbar.showMessage)

        self.board.start()
        self.center()

    def update(self):
        super.update()
        self.statusbar.showMessage(self.board.environ.nest.food)

    def center(self):

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2
        )


class Communicate(QtCore.QObject):
    msgToSB = QtCore.Signal(str)


class Board(QtWidgets.QFrame):

    BoardWidth = 50
    BoardHeight = 50
    Timer = 200

    def __init__(self, parent):
        super(Board, self).__init__()

        self.timer = QtCore.QBasicTimer()
        self.environ = Environment(h=Board.BoardHeight, w=Board.BoardWidth, agentType="DominAnt")

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.c = Communicate()

    def start(self):
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
