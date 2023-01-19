# from gcs_for_blocks.set_tesselation_2d import SetTesselation
from gcs_for_blocks.gcs_options import GCSforAutonomousBlocksOptions
from gcs_for_blocks.gcs_auto_blocks import GCSAutonomousBlocks
from gcs_for_blocks.util import timeit, INFO, all_possible_combinations_of_items
from gcs_for_blocks.relations import Relation, CenteredPolygon, EqPolygon, Box

import numpy as np

from pydrake.geometry.optimization import (  # pylint: disable=import-error
    Point,
    HPolyhedron,
    ConvexSet,
)

from draw_2d import Draw2DSolution

if __name__ == "__main__":
    # nb = 2
    # ubf = 2.0
    # start_point = Point( np.array([0,0, 2,2]))
    # target_point = Point(np.array([2,2, 0,0]))


    # nb = 3
    # ubf = 7.0
    # start_state = np.array([1,1, 6,6, 3.5,3.5])
    # target_state = np.array([6,6, 1,1, 3.5,3.5])

    # b1 = Box(width=1,height=1,t=0)
    # b2 = Box(width=1,height=1,t=1)



    nb = 2
    ubf = 4
    start_state = np.array([0,0, 4,2])
    target_state = np.array([4,2, 0,0])

    b1 = EqPolygon(num_sides=4, radius=0.499, t=0)
    b2 = EqPolygon(num_sides=4, radius=0.499, t=1)
    # b3 = Box(width=1,height=1, t=2)
    objects = [b1, b2]
    # objects = [b1, b2, b3]
    
    o1 = Box(width=0.999, height=0.999, t=-1, center=np.array([2,2])) # i must be doing something here wrong
    # o1 = EqPolygon(num_sides=3, radius=0.5, t=-1, center=np.array([2,2]))
    obstacles = [ o1 ]
    # obstacles = [ ]
    

    options = GCSforAutonomousBlocksOptions(num_blocks = nb, ubf=ubf)
    options.use_convex_relaxation = True
    options.max_rounded_paths = 100
    options.problem_complexity = "collision-free-all-moving"
    options.edge_gen = "frontier"  # binary_tree_down
    # options.lazy_set_construction = True
    # options.rounding_seed = 1
    # options.custom_rounding_paths = 0

    x = timeit()
    gcs = GCSAutonomousBlocks(options)
    gcs.init_things(obstacles, objects)
    gcs.init_start_target(start_state, target_state)

    gcs.solve(show_graph=True, verbose=True)

    modes, vertices = gcs.get_solution_path()

    drawer = Draw2DSolution(nb, np.array([ubf,ubf]), modes, vertices, target_state, fast = False, no_arm = True, draw_circles=True)
    drawer.draw_solution_one_by_one()
    # drawer.draw_solution_no_arm_obstacles(obstacles)