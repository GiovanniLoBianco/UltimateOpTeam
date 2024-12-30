from functools import cached_property
import os
import json
from pathlib import Path
from typing import Sequence

from ortools.linear_solver import pywraplp

from ..data.player import Player

M = 3


class UT_MILP_Model:
    """
    MILP model for the Ultimate Team problem.
    """

    def __init__(self, players: Sequence[Player], formation: str, alpha: float = 0.8):
        """
        Attributes
        ----------
        - players: Sequence[Player]
            sequence of players
        - formation: str
            formation name
        - alpha: float, default 0.8
            The weight of team chemistry in the objective function. The weight of team rating is 1 -
            alpha.

        """
        self.players = tuple(players)
        self.formation = formation
        self.alpha = alpha
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
        self._add_constraint()

        # Objective
        self._add_objective()

    @cached_property
    def positions(self) -> tuple[str]:
        path_formation = Path(
            os.environ["ULTIMATE_OPTEAM_PROJECT_PATH"]
            + "/src/ultimate_opteam/data/formations.json"
        )
        with open(path_formation, "r") as file:
            formations = json.load(file)
            return formations[self.formation]

    @cached_property
    def nation(self) -> tuple[str]:
        return tuple(set([player.nation for player in self.players]))

    @cached_property
    def league(self) -> tuple[str]:
        return tuple(set([player.league for player in self.players]))

    @cached_property
    def club(self) -> tuple[str]:
        return tuple(set([player.club for player in self.players]))

    def _declare_variables(self):
        # assignment variables
        for i_player, _ in enumerate(self.players):
            for k_pos, _ in enumerate(self.positions):
                self.x[(i_player, k_pos)] = self.solver.BoolVar(f"x_{i_player}_{k_pos}")

        # category coherence variables
        for cat in ["nation", "league", "club"]:
            self.y[cat] = {}
            for k_pos, _ in enumerate(self.positions):
                for j_cat, _ in enumerate(getattr(self, cat)):
                    self.y[cat][(k_pos, j_cat)] = self.solver.BoolVar(
                        f"y^{cat}_{k_pos}_{j_cat}"
                    )
        self.y["icon"] = {}
        for k_pos, _ in enumerate(self.positions):
            for j_nat, _ in enumerate(self.nation):
                self.y["icon"][(k_pos, j_nat)] = self.solver.BoolVar(
                    f"y^icon_{k_pos}_{j_nat}"
                )
        self.y["hero"] = {}
        for k_pos, _ in enumerate(self.positions):
            for j_league, _ in enumerate(self.league):
                self.y["hero"][(k_pos, j_league)] = self.solver.BoolVar(
                    f"y^hero_{k_pos}_{j_league}"
                )

        # score mode variables
        for cat in ["nation", "league", "club"]:
            self.gamma[cat] = {}
            for j_cat, _ in enumerate(getattr(self, cat)):
                for mode in range(4):
                    self.gamma[cat][(j_cat, mode)] = self.solver.BoolVar(
                        f"gamma^{cat}_{j_cat}_{mode}"
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
        self._add_players_assignment_constraint()
        self._add_category_coherence_constraint()
        self._add_score_mode_constraint()
        self._add_score_position_constraint()

    def _add_players_assignment_constraint(self):
        """
        Constraints related to players to positions assignment.
        """
        for k_pos, _ in enumerate(self.positions):
            self.solver.Add(
                self.solver.Sum(
                    self.x[(i_player, k_pos)] for i_player, _ in enumerate(self.players)
                )
                <= 1
            )

        for i_player, _ in enumerate(self.players):
            self.solver.Add(
                self.solver.Sum(
                    self.x[(i_player, k_pos)] for k_pos, _ in enumerate(self.positions)
                )
                <= 1
            )

        for i_player, player in enumerate(self.players):
            for k_pos, position in enumerate(self.positions):
                if not player.can_play_at(position):
                    self.solver.Add(self.x[(i_player, k_pos)] == 0)

    def _add_category_coherence_constraint(self):
        """
        Constraints related to category to position assignment.
        """
        for cat in ["nation", "league", "club"]:
            for k_pos, _ in enumerate(self.positions):
                for j_cat, _ in enumerate(getattr(self, cat)):
                    self.solver.Add(
                        self.solver.Sum(
                            self.x[(i_player, k_pos)]
                            for i_player, player in enumerate(self.players)
                            if getattr(player, cat) == getattr(self, cat)[j_cat]
                        )
                        >= self.y[cat][(k_pos, j_cat)]
                    )

        for k_pos, _ in enumerate(self.positions):
            for j_nat, _ in enumerate(self.nation):
                self.solver.Add(
                    self.solver.Sum(
                        self.x[(i_player, k_pos)]
                        for i_player, player in enumerate(self.players)
                        if player.nation == self.nation[j_nat]
                    )
                    >= self.y["icon"][(k_pos, j_nat)]
                )

        for k_pos, _ in enumerate(self.positions):
            for j_league, _ in enumerate(self.league):
                self.solver.Add(
                    self.solver.Sum(
                        self.x[(i_player, k_pos)]
                        for i_player, player in enumerate(self.players)
                        if player.league == self.league[j_league]
                    )
                    >= self.y["hero"][(k_pos, j_league)]
                )

    def _add_score_mode_constraint(self):
        """
        Constraints related to how score is computed for each nation, club and league.
        """

        for j_nation, _ in enumerate(self.nation):
            self.solver.Add(
                3 * self.gamma["nation"][(j_nation, 1)]
                + 5 * self.gamma["nation"][(j_nation, 2)]
                + 8 * self.gamma["nation"][(j_nation, 3)]
                <= self.solver.Sum(
                    self.y["nation"][(k_pos, j_nation)]
                    for k_pos, _ in enumerate(self.positions)
                )
                + self.solver.Sum(
                    self.y["icon"][(k_pos, j_nation)]
                    for k_pos, _ in enumerate(self.positions)
                )
            )
            self.solver.Add(
                self.solver.Sum(
                    self.gamma["nation"][(j_nation, mode)] for mode in range(4)
                )
                <= 1
            )

        for j_league, _ in enumerate(self.league):
            self.solver.Add(
                3 * self.gamma["league"][(j_league, 1)]
                + 5 * self.gamma["league"][(j_league, 2)]
                + 8 * self.gamma["league"][(j_league, 3)]
                <= self.solver.Sum(
                    self.y["league"][(k_pos, j_league)]
                    for k_pos, _ in enumerate(self.positions)
                )
                + self.solver.Sum(
                    self.y["hero"][(k_pos, j_league)]
                    for k_pos, _ in enumerate(self.positions)
                )
            )
            self.solver.Add(
                self.solver.Sum(
                    self.gamma["league"][(j_league, mode)] for mode in range(4)
                )
                <= 1
            )

        for j_club, _ in enumerate(self.club):
            self.solver.Add(
                2 * self.gamma["club"][(j_club, 1)]
                + 5 * self.gamma["club"][(j_club, 2)]
                + 8 * self.gamma["club"][(j_club, 3)]
                <= self.solver.Sum(
                    self.y["club"][(k_pos, j_club)]
                    for k_pos, _ in enumerate(self.positions)
                )
            )
            self.solver.Add(
                self.solver.Sum(self.gamma["club"][(j_club, mode)] for mode in range(4))
                <= 1
            )

        for i_player, player in enumerate(self.players):
            self.solver.Add(
                self.chemistry["player"][i_player]
                <= self.solver.Sum(
                    mode * self.gamma[cat][(j_cat, mode)]
                    for cat in ["nation", "league", "club"]
                    for j_cat, _ in enumerate(getattr(self, cat))
                    for mode in range(4)
                    if getattr(player, cat) == getattr(self, cat)[j_cat]
                )
            )

    def _add_score_position_constraint(self):
        """
        Constraints related to computation of score at each position.
        """
        for k_pos, _ in enumerate(self.positions):
            for i_player, player in enumerate(self.players):
                self.solver.Add(
                    self.chemistry["position"][k_pos]
                    <= (1 - self.x[(i_player, k_pos)]) * M
                    + self.chemistry["player"][i_player]
                )

            self.solver.Add(
                self.final_chemistry[k_pos]
                <= M
                * self.solver.Sum(
                    self.x[(i_player, k_pos)]
                    for i_player, player in enumerate(self.players)
                    if player.icon or player.hero
                )
                + self.chemistry["position"][k_pos]
            )

    def _add_objective(self):
        self.solver.Maximize(
            self.alpha
            * self.solver.Sum(
                self.final_chemistry[k_pos] for k_pos, _ in enumerate(self.positions)
            )
            + (1 - self.alpha)
            * self.solver.Sum(
                self.x[(i_player, k_pos)] * player.rating
                for i_player, player in enumerate(self.players)
                for k_pos, _ in enumerate(self.positions)
            )
        )

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
