# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

try:
    from tkinter import Tk, Canvas, Toplevel
except ImportError:
    from Tkinter import Tk, Canvas, Toplevel

import time

# colors
BLOCK_COLORS = ["#E3B5A4", "#E8D6CB", "#C3DFE0", "#F6E4F6", "#F4F4F4"]
ARM_COLOR = "#843B62"
ARM_NOT_EMPTY_COLOR = "#5E3886"  # 5E3886 621940
TEXT_COLOR = "#0B032D"
BLACK = "#0B032D"
BACKGROUND = "#F5E9E2"

# sizes
CELL_WIDTH = 90
BLOCK_SIZE = CELL_WIDTH-2
ARM_SIZE = CELL_WIDTH
BORDER_WIDTH = 20
PADDING_WIDTH = CELL_WIDTH / 2

# timing and speeds
MOVE_DT = 0.033  # s
FAST_SPEED = 8 # units/s
FAST_GRASP_DT = 0.3  # s
SLOW_SPEED = 1.5  # units/s
SLOW_GRASP_DT = 0.5  # s

def get_pixel_location(loc):
    return (
        loc[0] * CELL_WIDTH + PADDING_WIDTH + BORDER_WIDTH,
        loc[1] * CELL_WIDTH + PADDING_WIDTH + BORDER_WIDTH,
    )

# to draw blocks and circle
class ThingToDraw:
    def __init__(self, canvas, cells, name = "", fill_color = BLACK, width = None, height = None, draw_function = None, vertices = [], outline_color=BLACK)
        self.width = width
        self.height = height
        self.name = name
        self.outline_color = outline_color
        self.fill_color = fill_color
        self.canvas = canvas
        if draw_function is None:
            self.draw_function = self.canvas.create_rectangle
        else:
            self.draw_function = draw_function
        self.cells = cells
        self.vertices = np.array([list(get_pixel_location(loc)) for loc in vertices]) # nx2

    def draw_shape(self, block_state):
        x, y = get_pixel_location(block_state)
        side = BLOCK_SIZE / 2.0
        self.cells[(x, y)] = [
            self.draw_function(
                x - side,
                y - side,
                x + side,
                y + side,
                fill=self.fill_color,
                outline=self.outline_color,
                width=2,
            ),
            self.canvas.create_text(x, y, text=self.name, fill=TEXT_COLOR),
        ]

    def draw_polygon(self, block_state):
        x, y = get_pixel_location(block_state)
        self.cells[(x, y)] = [
            self.canvas.create_polygon(
                list((self.vertices + np.array([x, y])).flatten()),
                fill=self.fill_color,
                outline=self.outline_color,
                width=2,
            ),
            self.canvas.create_text(x, y, text=self.name, fill=TEXT_COLOR),
        ]

    def draw_line(self):
        self.cells[(10,10)] = [
            self.canvas.create_line(self.vertices, outline=self.fill_color, width=2),
        ]
        
 
class StaticThing(ThingToDraw):
    def __init__(self, canvas, cells, thing_type, name = "", fill_color = BLACK, block_state = None, width = None, height = None, draw_function = None, vertices = [], outline_color=BLACK):
        self.block_state = block_state
        self.thing_type = thing_type
        self.done = True
        super(StaticThing, self).__init__(canvas=canvas, cells=cells, name = name, fill_color = fill_color, width = width, height = height, draw_function = draw_function, vertices = vertices, outline_color=outline_color)

    def draw(self):
        if self.thing_type == "shape":
            self.draw_shape(self.block_state)
        elif self.thing_type == "polygon":
            self.draw_polygon(self.block_state)
        elif self.thing_type == "line":
            self.draw_line()

class AutonomousThing(ThingToDraw):
    def __init__(self, trajectory, speed, canvas, cells, thing_type, name = "", fill_color = BLACK, block_state = None, width = None, height = None, draw_function = None, vertices = [], outline_color=BLACK)
        self.trajectory = trajectory
        self.speed = speed
        self.distance_per_dt = self.speed * MOVE_DT
        self.state_now = self.trajectory[0, :] # n x 2
        self.index_next = 1
        self.thing_type = thing_type
        self.done = False
        super(AutonomousThing, self).__init__(canvas=canvas, cells=cells, name = name, fill_color = fill_color, width = width, height = height, draw_function = draw_function, vertices = vertices, outline_color=outline_color)

    def advance_along_trajectory(self):
        if self.index_next == len(self.trajectory):
            self.done = True
            return
        state_next = self.trajectory[self.index_next]
        delta = state_next - self.state_now
        distance = np.linalg.norm(delta)
        if distance / self.distance_per_dt < 1:
            self.state_now = state_next
            self.index_next += 1
        else:
            self.state_now += delta * self.distance_per_dt / distance

    def draw(self):
        if self.thing_type == "shape":
            self.draw_shape(self.state_now)
        elif self.thing_type == "polygon":
            self.draw_polygon(self.state_now)
        self.advance_along_trajectory()




class Draw2DSolution:
    def __init__(
        self,
        ub: npt.NDArray, # upper bound array
        fast: bool = True,
        no_padding=False,
    ):
        self.ub = ub

        if fast:
            self.speed, self.grasp_dt = FAST_SPEED, FAST_GRASP_DT
        else:
            self.speed, self.grasp_dt = SLOW_SPEED, SLOW_GRASP_DT

        if no_padding:
            PADDING_WIDTH = 0

        self.width = CELL_WIDTH * self.ub + PADDING_WIDTH * 2 + BORDER_WIDTH * 2

        # tkinter initialization
        self.tk = Tk()
        self.tk.withdraw()
        top = Toplevel(self.tk)
        top.wm_title("Moving Blocks")
        top.protocol("WM_DELETE_WINDOW", top.destroy)

        self.canvas = Canvas(top, width=self.width[0], height=self.width[1], background=BACKGROUND)
        self.canvas.pack()
        self.cells = {}
        self.environment = []

        self.things = []

    def draw_things(self):
        done = True
        for thing in self.things:
            thing.draw()
            if not thing.done:
                done = False        
        return done

    def draw_solution_no_arm(self):

        # draw initial state
        done = self.draw()
        time.sleep(2.0)

        # draw state by state
        while not done:
            done = self.draw()
            time.sleep(MOVE_DT)
        time.sleep(2.0)

    def draw(self):
        self.clear()
        done = self.draw_background()
        self.draw_things()
        self.tk.update()
        return done

    def clear(self):
        self.canvas.delete("all")

    def add_goal(self, goal):
        rgoal = np.reshape( (int(len(goal)/2), 2) )
        for i in range(rgoal):
            self.add_thing()

    def draw_shadow(self, state, name):
        x, y = get_pixel_location(state)
        side = BLOCK_SIZE / 2.0
        if name == "arm" or self.draw_circles:
            create_func = self.canvas.create_oval
        else:
            create_func = self.canvas.create_rectangle
        self.cells[(x, y)] = [
            create_func(
                x - side,
                y - side,
                x + side,
                y + side,
                fill="#D3D3D3",
                outline="grey",
                width=2,
            ),
            self.canvas.create_text(x, y, text=name, fill=TEXT_COLOR),
        ]

    def draw_background(self):
        self.environment.append(
            [
                self.canvas.create_rectangle(
                    0,
                    0,
                    BORDER_WIDTH,
                    self.width[1],
                    fill="black",
                    outline="black",
                    width=0,
                ),
                self.canvas.create_rectangle(
                    0,
                    0,
                    self.width[0],
                    BORDER_WIDTH,
                    fill="black",
                    outline="black",
                    width=0,
                ),
                self.canvas.create_rectangle(
                    0,
                    self.width[1] - BORDER_WIDTH,
                    self.width[0],
                    self.width[1],
                    fill="black",
                    outline="black",
                    width=0,
                ),
                self.canvas.create_rectangle(
                    self.width[0] - BORDER_WIDTH,
                    0,
                    self.width[0],
                    self.width[1],
                    fill="black",
                    outline="black",
                    width=0,
                ),
            ]
        )

    def draw_goal(self):
        if self.no_arm:
            for i in range(self.num_modes):
                self.draw_shadow(self.goal[2 * i : 2 * i + 2], i)
        else:
            self.draw_shadow(self.goal[0:2], "arm")
            for i in range(1, self.num_modes):
                self.draw_shadow(self.goal[2 * i : 2 * i + 2], i)
