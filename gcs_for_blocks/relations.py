#!/usr/bin/env python3
# pyright: reportMissingImports=false
import typing as T

import numpy as np
import numpy.typing as npt

from .polygons import CenteredPolygon, EqPolygon, Box

from pydrake.geometry.optimization import HPolyhedron

from .gcs_options import GCSforAutonomousBlocksOptions
from .util import (
    WARN,
    INFO,
    all_possible_combinations_of_items,
    timeit,
    ChebyshevCenter,
    normalized,
    perp_left,
    perp_right
)

from tqdm import tqdm


from scipy.spatial import ConvexHull

class Relation:
    def __init__(self, poly1:CenteredPolygon, poly2:CenteredPolygon, name: str = ""):
        self.poly1 = poly1
        self.poly2 = poly2
        self.name = name
        assert self.poly1.t >= 0, "first poly in a relation must be an object"

        # get outer side
        minkowski_vertex_sum = np.array([(poly1.vertices[i, :] - poly2.vertices[j, :]) for i in range(poly1.n) for j in range(poly2.n)])
        hull = ConvexHull(minkowski_vertex_sum)
        self.v = minkowski_vertex_sum[hull.vertices, :]
        self.n = len(self.v)

    def get_dir_nbhd(self, dir:int):
        assert 0 <= dir < self.n
        return [ (dir-1)%self.n, (dir+1)%self.n ]

    def get_convex_set_for_direction(self, dir: int, state_dim:int) -> T.Tuple[npt.NDArray, npt.NDArray]:
        bd = 2
        sd = state_dim
        # init rows
        a0, a1, a2 = np.zeros(sd), np.zeros(sd), np.zeros(sd)

        L, R = self.v[dir, :], self.v[ (dir+1)%self.n, :]
        Rl = perp_left(L)
        Rp = perp_right(R)
        # Rl = normalized(perp_left(L))
        # Rp = normalized(perp_right(R))
        P = R - L
        Pp = perp_right(P)
        # Pp = normalized(perp_right(P))
        
        # poly1 is always object
        x11, x12 = self.poly1.t * bd, self.poly1.t * bd + 1
        # if poly2 is object
        if self.poly2.t >= 0:
            x21, x22 = self.poly2.t * bd, self.poly2.t * bd + 1
            b = np.array([0, 0, - Pp.dot(L+R)/2])
            a0[x11: x12+1] = Rl
            a1[x11: x12+1] = Rp
            a2[x11: x12+1] = Pp

            a0[x21: x22+1] = -Rl
            a1[x21: x22+1] = -Rp
            a2[x21: x22+1] = -Pp
        else:
            # poly2 is an obstacle -- results in scalars 
            x2 = self.poly2.center
            b0 = Rl.dot(x2)
            b1 = Rp.dot(x2)
            b2 = Pp.dot(x2) - Pp.dot(L+R)/2
            b = np.array([b0,b1,b2])
            a0[x11: x12+1] = Rl
            a1[x11: x12+1] = Rp
            a2[x11: x12+1] = Pp

        A = np.vstack((a0, a1, a2))
        return A, b

    def get_direction_for_point(self, x: npt.NDArray):
        for dir in range(self.n):
            A, b = self.get_convex_set_for_direction(dir, len(x))
            if np.all(A.dot(x) <= b):
                return dir
        assert False, "Provided point is part of no set"


class TesselationMaster:
    def __init__(self, relations:T.List[Relation], opt):
        self.relations = relations
        self.rel_len = len(self.relations)
        self.opt = opt

    def get_grounded_relation_for_point(self, x: npt.NDArray):
        return [rel.get_direction_for_point(x) for rel in self.relations]
    
    def get_set_for_name(self, rel_name:str):
        return self.get_set_for_grounded_relation(self.name_to_rel(rel_name))
    
    def get_set_for_grounded_relation(self, grounded_relation: T.List[int]) -> HPolyhedron:
        A, b = self.get_bounding_box_constraint()
        for i, dir in enumerate(grounded_relation):
            A_relation, b_relation = self.relations[i].get_convex_set_for_direction(dir, self.opt.state_dim)
            A = np.vstack((A, A_relation))
            b = np.hstack((b, b_relation))
        return HPolyhedron(A, b)

    def get_bounding_box_constraint(self) -> T.Tuple[npt.NDArray, npt.NDArray]:
        A = np.vstack((np.eye(self.opt.state_dim), -np.eye(self.opt.state_dim)))
        b = np.hstack((self.opt.ub, -self.opt.lb))
        return A, b

    def name(self, grounded_relation:T.List[int]):
        assert len(grounded_relation) == self.rel_len
        return "_".join([str(x) for x in grounded_relation])
    
    def name_to_rel(self, rel_name:str):
        rel = [int(x) for x in rel_name.split("_")]
        assert len(rel) == self.rel_len
        return rel
    
    def get_1_step_neighbours(self, rel_name: str):
        """
        Get all 1 step neighbours
        1-step -- change of a single relation
        """
        grounded_relation = self.name_to_rel(rel_name)
        nbhd = []
        # print(grounded_relation)
        for i, dir in enumerate(grounded_relation):
            nbh_grounded_relation = grounded_relation.copy()
            nbh_directions = self.relations[i].get_dir_nbhd(dir)
            for nbh_dir in nbh_directions:
                nbh_grounded_relation[i] = nbh_dir
                # TODO: do i check here whether proposed set is empty?
                nbhd += [self.name(nbh_grounded_relation)]
        # print(nbhd)
        return nbhd
    
    def get_useful_1_step_neighbours(self, rel_name: str, target_name: str):
        """
        Get 1-stop neighbours that are relevant given the target node
        1-step -- change in a single relation
        relevant to target -- if relation in relation is already same as in target, don't change it
        """
        grounded_relation = self.name_to_rel(rel_name)
        grounded_target = self.name_to_rel(target_name)
        # print(grounded_relation)

        nbhd = []
        for i, dir in enumerate(grounded_relation):
            # same direction along this relation -- continue
            if dir == grounded_target[i]:
                continue

            relation = self.relations[i]

            # nbh_grounded_relation = grounded_relation
            # for nbh_dir in relation.get_dir_nbhd(dir):
            #     nbh_grounded_relation[i] = nbh_dir
            #     nbhd += [ nbh_grounded_relation ]


            up_dist = (grounded_target[i] - grounded_relation[i]) % relation.n
            down_dist = (grounded_relation[i] - grounded_target[i]) % relation.n

            nbh_grounded_relation = grounded_relation.copy()
            nbh_relations = relation.get_dir_nbhd(dir)

            if up_dist == down_dist:
                for nbh_dir in relation.get_dir_nbhd(dir):
                    nbh_grounded_relation[i] = nbh_dir
                    nbhd += [ self.name(nbh_grounded_relation) ]
            elif up_dist < down_dist:
                nbh_grounded_relation[i] = nbh_relations[1]
                nbhd += [ self.name(nbh_grounded_relation) ]
            else:
                nbh_grounded_relation[i] = nbh_relations[0]
                nbhd += [ self.name(nbh_grounded_relation) ]
        # print(nbhd)
        # assert False
        return nbhd