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

# results: 4 element list: [best chromosome at each epoch,
#                           full distribution of scores at each, 
#                           all 800 chromosomes of final epoch, 
#                           food]

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
        # self.chrom_input.setText(
        #     "C:/Users/evere/Documents/CornellTech/SwarmRobotics/Ant-Neuroevolution/results.pkl"
        # )
        self.chrom_input.setPlaceholderText("Chromosome")

        self.file_button = QtWidgets.QPushButton("Select file")
        self.file_button.clicked.connect(self.select_file)

        self.epoch_label = QtWidgets.QLabel('Epoch:')
        self.epoch_input = QtWidgets.QLineEdit()
        self.epoch_input.setText('-1')

        self.score_label = QtWidgets.QLabel('nth best score:')
        self.score_input = QtWidgets.QLineEdit()
        self.score_input.setText('0')

        self.hide_ants_button = QtWidgets.QPushButton('Hide Ants')
        self.hide_ants_button.clicked.connect(self.board.toggle_ants)

        self.start_button = QtWidgets.QPushButton('Reset')
        self.start_button.clicked.connect(self.start)

        self.pause_button = QtWidgets.QPushButton('>||')
        self.pause_button.clicked.connect(self.board.pause)

        self.control_layout = QtWidgets.QHBoxLayout()
        self.control_layout.addWidget(self.chrom_input)
        self.control_layout.addWidget(self.file_button)
        self.control_layout.addWidget(self.epoch_label)
        self.control_layout.addWidget(self.epoch_input)
        self.control_layout.addWidget(self.score_label)
        self.control_layout.addWidget(self.score_input)
        self.control_layout.addWidget(self.hide_ants_button)
        self.control_layout.addWidget(self.start_button)
        self.control_layout.addWidget(self.pause_button)

        self.controls = QtWidgets.QWidget()
        self.controls.setLayout(self.control_layout)

        self.dock = QtWidgets.QDockWidget("Controls", self)
        self.dock.setFloating(True)
        self.dock.setWidget(self.controls)

        # self.board.start()
        self.center()

    def update(self):
        super.update()
        self.statusbar.showMessage(self.board.environ.nest.food)

    def start(self):
        self.isStarted = True
        chrom_file = self.chrom_input.text()

        # If there's a chromosome file, use it
        if len(chrom_file) > 0:
            epoch_n = int(self.epoch_input.text())
            order_n = int(self.score_input.text())
            pickle_off = open(chrom_file, "rb")
            temp = pickle.load(pickle_off)
            chroms = np.array(temp[0][epoch_n])
            scores = np.array(temp[1][epoch_n]).argsort()
            chrom = chroms[epoch_n]#[scores[order_n]]

            chrom = get_best(temp)
            self.board.start(chrom)

        # Otherwise, run determineant
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

    def mousePressEvent(self, QMouseEvent):
        self.board.addFood(QMouseEvent.pos())


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
        self.isStarted = False
        self.isPaused = True
        self.show_ants = True

    def start(self, chromosome=None):
        self.isStarted = True
        self.environ = Environment(chromosome)
        # self.draw_checkers()
        self.timer = QtCore.QBasicTimer()
        self.update()
    
    def pause(self):
        if not self.isStarted:
            return
        self.isPaused = not self.isPaused
        if self.isPaused:
            self.timer.stop()
        else:
            self.timer.start(Board.Timer, self)
    
    def toggle_ants(self):
        self.show_ants = not self.show_ants

    def draw_checkers(self):
        for i in range(Board.BoardWidth * Board.BoardHeight):
            if i % 2 == 0:
                row = i//Board.BoardHeight
                col = i%Board.BoardHeight
                self.environ.grid[row][col].food = 1

    def squareWidth(self):
        return self.contentsRect().width() // Board.BoardWidth

    def squareHeight(self):
        return self.contentsRect().height() // Board.BoardHeight
    
    def addFood(self, pos):
        col = pos.x()//self.squareWidth()
        row = pos.y()//self.squareHeight()
        self.environ.grid[row][col].food += 1
        self.update()

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
                    boardTop + row  * self.squareHeight(),
                    cell,
                )
        for agent in self.environ.agents:
            has_food = agent.has_food
            row, col = agent.get_coord()
            self.drawSquare(
                painter,
                rect.left() + col * self.squareWidth(),
                boardTop + row  * self.squareHeight(),
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
        color = None
        if self.show_ants and ant:
            color = QtGui.QColor.fromHsv(*Colors.ANT)
        if cell:
            if cell.is_nest:
                color = QtGui.QColor.fromHsv(*Colors.NEST)
            elif cell.pheromone > 0:  # draw pheromone
                h,s,v = Colors.PHER
                if Colors.mode == 'dark':
                    v = v*min(cell.pheromone,1)
                else:
                    s = s * min(cell.pheromone,1)
                color = QtGui.QColor.fromHsv(h, s, v) 
            elif cell.pheromone == 0: 
                color = QtGui.QColor.fromHsv(*Colors.FREE)
            elif cell.pheromone <0:
                color = QtGui.QColor(0xFFC0CB)
        if not color:
            color = QtGui.QColor.fromHsv(*Colors.FREE)
        padding = 1
        if Colors.mode == 'dark':
            padding = 0
        painter.fillRect(
            x+padding, y+padding,
            self.squareWidth() - padding, self.squareHeight() -padding,
            color
        )

        # Draw food
        if (cell and not cell.is_nest and cell.food > 0) or (ant and ant.has_food):  
            color = QtGui.QColor.fromHsv(*Colors.FOOD)
            painter.fillRect(
            x + self.squareWidth()//4+1, y + self.squareHeight()//4+1, 
            self.squareWidth()//2, self.squareHeight()//2, color
        )

class Colors(object):
    '''
    c stores colors in hsv
    '''
    # Bright mode
    mode = 'bright'
    c = [
        (0, 0, 255), # White space
        (0, 255, 255), # Red Ant
        (50, 100, 20), # brown nest
        (233, 255, 255), # blue pheromone
        (112, 255, 255) # Green food
    ]

    # dark mode
    # mode = 'dark'
    # c = [
    #     (0, 0, 0), # black space
    #     (0, 255, 255), # Red Ant
    #     (26, 255, 200), # brown nest
    #     (233, 255, 255), # blue pheromone
    #     (112, 255, 255) # Green food
    # ]

    FREE = c[0]
    ANT =  c[1]
    NEST = c[2]
    PHER = c[3]
    FOOD = c[4]

def main():

    app = QtWidgets.QApplication(sys.argv)
    t = AntGUI()
    t.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
