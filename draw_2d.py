# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

# from test import (
#     make_some_simple_transparent_tests,
#     make_simple_obstacle_swap_two,
#     make_simple_transparent_gcs_test,
#     make_simple_obstacle_swap_two_in_out,
# )

try:
    from tkinter import Tk, Canvas, Toplevel
except ImportError:
    from Tkinter import Tk, Canvas, Toplevel


# import multiprocessing
# import Tkinter as tk
# import cv2

import time

BLOCK_COLORS = ["#E3B5A4", "#E8D6CB", "#C3DFE0", "#F6E4F6", "#F4F4F4", "#000000"]
# BLOCK_COLORS = ["#000000", "#E8D6CB", "#C3DFE0", "#F6E4F6", "#F4F4F4"]
ARM_COLOR = "#843B62"
ARM_NOT_EMPTY_COLOR = "#5E3886"  # 5E3886 621940

TEXT_COLOR = "#0B032D"
BLACK = "#0B032D"
BACKGROUND = "#F5E9E2"
CELL_WIDTH = 50
CELL_WIDTH = 90

class Trajectory:

    def __init__(self, trajectory, speed, move_dt):
        self.move_dt = move_dt
        self.trajectory = trajectory
        self.speed = speed
        self.now_index = 0
        self.now = self.trajectory[self.now_index, :]

    def get_next(self):
        if self.now_index == len(self.trajectory):
            return self.now
        else:
            state_now = self.now
            state_next = self.trajectory[self.now_index+1,:]
            delta = state_next - state_now
            distance = np.linalg.norm(delta)
            distance_per_dt = self.speed * self.move_dt
            if float(distance / distance_per_dt) < 1:
                self.now = state_next
                self.now_index += 1
            else:
                num_steps = int(max(float(distance / distance_per_dt), 1))
                self.now = state_now + delta * 1 / num_steps
            print(self.now)
            return self.now




class Draw2DSolution:
    def __init__(
        self,
        num_modes: int,
        ub: npt.NDArray,
        mode_solution: T.List[str],
        vertex_solution: T.List[T.List],
        goal,
        fast: bool = True,
        no_arm=False,
        no_padding=False,
        draw_circles = False,
    ):
        self.no_arm = no_arm
        self.num_modes = num_modes
        self.ub = ub
        self.cell_scaling = CELL_WIDTH
        self.block_size = CELL_WIDTH-2
        self.arm_size = CELL_WIDTH

        self.padding = self.block_size / 2
        if no_padding:
            self.padding = 0
        self.border = 20

        self.mode = mode_solution
        self.vertex = vertex_solution

        if fast:
            self.speed = 8  # units/s
            self.grasp_dt = 0.3  # s
        else:
            self.speed = 1.5  # units/s
            self.grasp_dt = 0.5  # s

        self.move_dt = 0.033  # s
        self.obstacles = None
        

        self.grasping = False
        self.draw_circles = draw_circles

        self.width = self.cell_scaling * self.ub + self.padding * 2 + self.border * 2
        self.goal = goal

        # tkinter initialization
        self.tk = Tk()
        self.tk.withdraw()
        top = Toplevel(self.tk)
        top.wm_title("Moving Blocks")
        top.protocol("WM_DELETE_WINDOW", top.destroy)

        self.canvas = Canvas(top, width=self.width[0], height=self.width[1], background=BACKGROUND)
        # print(self.width)
        self.canvas.pack()
        self.cells = {}
        self.environment = []

    def get_pixel_location(self, loc):
        return (
            loc[0] * self.cell_scaling + self.padding + self.border,
            loc[1] * self.cell_scaling + self.padding + self.border,
        )

    def draw_solution_no_arm(self):
        vertex = self.vertex
        state_now = vertex[0, :]  # type: ignore

        # draw initial state
        self.draw_state(state_now)
        time.sleep(2.0)

        for i in range(1, len(vertex)):
            state_next = vertex[i, :]  # type: ignore
            self.move_from_to(state_now, state_next)
            state_now = state_next
        time.sleep(2.0)

    def draw_solution_no_arm_obstacles(self, obstacles):
        vertex = self.vertex
        state_now = vertex[0, :]  # type: ignore

        self.obstacles = obstacles

        # draw initial state
        self.draw_state(state_now)
        time.sleep(2.0)

        for i in range(1, len(vertex)):
            state_next = vertex[i, :]  # type: ignore
            self.move_from_to(state_now, state_next)
            state_now = state_next
        time.sleep(2.0)

    

    def draw_solution_one_by_one(self):
        vertex = self.vertex
        mode = self.mode

        self.trajectories = [ Trajectory(vertex[:, 2*i:2*i+2], self.speed, self.move_dt) for i in range(self.num_modes) ]
        print(self.trajectories)
        self.trajs = [ vertex[:, 2*i:2*i+2] for i in range(self.num_modes) ]

        # def __init__(self, trajectory, speed, move_dt):

        print(self.trajectories)
        state_now = []
        state_done = []
        for i in range(len(self.trajs)):
            state_now += [self.trajs[i][0]]
            state_done += [self.trajs[i][-1]]

        self.draw_state(state_now)
        time.sleep(1.0)

        # while not np.allclose(state_now, state_done):

        print(state_now, state_done)
        while not np.allclose(state_now, state_done):
            next_state = np.array([traj.get_next() for traj in self.trajectories])
            print(next_state[0])
            next_state = next_state.flatten()
            # print(next_state)
            self.draw_state(next_state)
            time.sleep(self.move_dt)

        time.sleep(1.0)

    def draw_solution(self):
        vertex = self.vertex
        mode = self.mode
        state_now = vertex[0, :]  # type: ignore
        mode_now = mode[1]
        if mode_now == "0":
            self.grasping = False
        # draw initial state
        self.draw_state(state_now)
        time.sleep(1.0)

        for i in range(1, len(vertex)):
            state_next = vertex[i, :]  # type: ignore
            mode_next = mode[i]

            self.move_from_to(state_now, state_next)
            if mode_now != mode_next:
                if mode_next == "target":
                    self.draw_state(state_next)
                    time.sleep(2.0)
                elif mode_next != "0":
                    self.grasping = True
                else:
                    self.grasping = False
                self.grasp(state_next)
            mode_now = mode_next
            state_now = state_next
        time.sleep(1.0)

    def move_from_to(self, state_now, state_next):
        delta = state_next - state_now
        distance = np.linalg.norm(delta[0:2])
        if self.no_arm:
            distance = np.linalg.norm(delta)
        distance_per_dt = self.speed * self.move_dt
        num_steps = int(max(float(distance / distance_per_dt), 1.0))
        for i in range(1, num_steps + 1):
            # self.draw_traj()
            self.draw_state(state_now + delta * i / num_steps)
            time.sleep(self.move_dt)

    def grasp(self, state):
        self.draw_state(state)
        # time.sleep(self.grasp_dt)

    def draw_state(self, state):
        self.clear()
        self.draw_background()
        self.draw_goal()
        if self.no_arm:
            for i in range(self.num_modes):
                self.draw_block(state[2 * i : 2 * i + 2], i)
        else:
            for i in range(1, self.num_modes):
                self.draw_block(state[2 * i : 2 * i + 2], i)
            self.draw_arm(state[0:2])
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                self.draw_block(obstacle.center, 5, draw_function = self.canvas.create_rectangle)
        self.tk.update()

    def clear(self):
        self.canvas.delete("all")

    def draw_block(self, block_state, block_num, draw_function = None):
        x, y = self.get_pixel_location(block_state)
        side = self.block_size / 2.0

        if draw_function is None:
            if self.draw_circles:
                draw_function = self.canvas.create_oval
            else:
                draw_function = self.canvas.create_rectangle

        self.cells[(x, y)] = [
            draw_function(
                x - side,
                y - side,
                x + side,
                y + side,
                fill=BLOCK_COLORS[(block_num) % len(BLOCK_COLORS)],
                outline="black",
                width=2,
            ),
            self.canvas.create_text(x, y, text=block_num, fill=TEXT_COLOR),
        ]

    def draw_arm(self, arm_state):
        x, y = self.get_pixel_location(arm_state)
        side = self.arm_size / 2.0
        if self.grasping:
            arm_color = ARM_NOT_EMPTY_COLOR
        else:
            arm_color = ARM_COLOR

        self.cells[(x, y)] = [
            self.canvas.create_oval(
                x - side,
                y - side,
                x + side,
                y + side,
                fill=arm_color,
                outline="black",
                width=2,
            ),
            self.canvas.create_text(x, y, text="arm", fill=TEXT_COLOR),
        ]

    def draw_shadow(self, state, name):
        x, y = self.get_pixel_location(state)
        side = self.block_size / 2.0
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
                    self.border,
                    self.width[1],
                    fill="black",
                    outline="black",
                    width=0,
                ),
                self.canvas.create_rectangle(
                    0,
                    0,
                    self.width[0],
                    self.border,
                    fill="black",
                    outline="black",
                    width=0,
                ),
                self.canvas.create_rectangle(
                    0,
                    self.width[1] - self.border,
                    self.width[0],
                    self.width[1],
                    fill="black",
                    outline="black",
                    width=0,
                ),
                self.canvas.create_rectangle(
                    self.width[0] - self.border,
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
            # print("oi")
            # print(self.goal[0:2])
            self.draw_shadow(self.goal[0:2], "arm")
            for i in range(1, self.num_modes):
                self.draw_shadow(self.goal[2 * i : 2 * i + 2], i)


# if __name__ == "__main__":
#     gcs, ub, goal = make_simple_obstacle_swap_two()
#     # gcs, ub, goal = make_simple_obstacle_swap_two_in_out()

#     # gcs, ub, goal = make_simple_transparent_gcs_test(2, 7, 18)

#     # gcs, ub, goal = make_simple_transparent_gcs_test(2,10,21, use_convex_relaxation=False, display_graph=False, max_rounded_paths=0, add_grasp_cost = False)

#     # nb = 2
#     # h = 7
#     # seed = 5
#     # gcs, ub, goal = make_simple_transparent_gcs_test(
#     #     2,
#     #     nb,
#     #     h,
#     #     use_convex_relaxation=False,
#     #     display_graph=False,
#     #     max_rounded_paths=0,
#     #     add_grasp_cost=False,
#     #     randomize=False,
#     #     seed=seed,
#     # )

#     assert gcs.solution.is_success(), "Solution was not found"
#     modes, vertices = gcs.get_solution_path()
#     for i in range(len(vertices)):
#         vertices[i] = ["%.1f" % v for v in vertices[i]]

#     print(modes)
#     print(vertices)

#     drawer = Draw2DSolution(gcs.opt.num_modes, ub, modes, vertices, goal, fast=False)  # type: ignore
#     drawer.draw_solution()

#     # make_some_simple_transparent_tests()
