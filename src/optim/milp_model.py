from functools import cached_property
import os
import json
from pathlib import Path
from typing import Sequence

from ortools.linear_solver import pywraplp

from data.player import Player


class UT_MILP_Model:
    """
    MILP model for the Ultimate Team problem.

    Attributes
    ----------
    - players: list of players
    - formation: formation name
    """

    def __init__(self, players: Sequence[Player], formation: str):
        self.players = tuple(players)
        self.formation = formation
        self.solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")

        # Variables
        self.x = {}
        self.y = {}
        self.gamma = {}
        self.chemistry = {}
        self.final_chemistry = {}
        self._declare_variables()

        # Constraints
        self.constraints = []

    @cached_property
    def positions(self) -> tuple[str]:
        path_formation = Path(
            os.environ["ULTIMATE_OPTEAM_PROJECT_PATH"] + "src/data/formations.json"
        )
        with open(path_formation, "r") as file:
            formations = json.load(file)
            return formations[self.formation]

    @cached_property
    def nations(self) -> tuple[str]:
        return tuple([player.nation for player in self.players])

    @cached_property
    def leagues(self) -> tuple[str]:
        return tuple([player.league for player in self.players])

    @cached_property
    def clubs(self) -> tuple[str]:
        return tuple([player.club for player in self.players])

    def _declare_variables(self):
        # assignment variables
        for i_player, _ in enumerate(self.players):
            for k_pos, _ in enumerate(self.positions):
                self.x[(i_player, k_pos)] = self.solver.BoolVar(
                    0, 1, f"x_{i_player}_{k_pos}"
                )

        # category coherence variables
        for cat in ["nations", "leagues", "clubs"]:
            self.y[cat] = {}
            for k_pos, _ in enumerate(self.positions):
                for j_cat, _ in enumerate(getattr(self, cat)):
                    self.y[cat][(k_pos, j_cat)] = self.solver.BoolVar(
                        0, 1, f"y^{cat}_{k_pos}_{j_cat}"
                    )
        self.y["icon"] = {}
        for k_pos, _ in enumerate(self.positions):
            for j_nat, _ in enumerate(self.nations):
                self.y["icon"][(k_pos, j_nat)] = self.solver.BoolVar(
                    0, 1, f"y^icon_{k_pos}_{j_nat}"
                )
        self.y["hero"] = {}
        for k_pos, _ in enumerate(self.positions):
            for j_league, _ in enumerate(self.leagues):
                self.y["hero"][(k_pos, j_league)] = self.solver.BoolVar(
                    0, 1, f"y^hero_{k_pos}_{j_league}"
                )

        # score mode variables
        for cat in ["nations", "leagues", "clubs"]:
            self.gamma[cat] = {}
            for j_cat, _ in enumerate(getattr(self, cat)):
                for mode in range(4):
                    self.gamma[cat][(j_cat, mode)] = self.solver.BoolVar(
                        0, 1, f"gamma^{cat}_{j_cat}_{mode}"
                    )

        # chemistry variables
        self.chemistry["player"] = {}
        for i_player, _ in enumerate(self.players):
            self.chemistry["player"][i_player] = self.solver.IntVar(
                0, 3, f"ch^player_{i_player}"
            )
        self.chemistry["position"] = {}
        for k_pos, _ in enumerate(self.positions):
            self.chemistry["position"][k_pos] = self.solver.IntVar(
                0, 3, f"ch^position_{k_pos}"
            )
        for k_pos, _ in enumerate(self.positions):
            self.final_chemistry[k_pos] = self.solver.IntVar(0, 3, f"final_ch_{k_pos}")

    def _add_constraint(self):
        pass

    def solve(self):
        for constraint in self.constraints:
            self.solver.Add(constraint)

        status = self.solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            return self.solver.Objective().Value()
        else:
            return None

    def get_variable_value(self, name):
        return self.variables[name].solution_value()

    def get_solver(self):
        return self.solver

    def get_objective(self):
        return self.solver.Objective()

    def get_objective_value(self):
        return self.solver.Objective().Value()

    def get_solution(self):
        return {name: self.get_variable_value(name) for name in self.variables.keys()}
