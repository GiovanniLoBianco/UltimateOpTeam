from functools import cached_property
import os
import json
from pathlib import Path
from typing import Sequence
import loguru

from ortools.linear_solver import pywraplp
import numpy as np

from ..data import Player, Team

M = 3

logger = loguru.logger()


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

    def _extract_team_from_solution(self):
        """
        Extract team from solver's current solution.
        """
        composition = []
        for k_pos, position in enumerate(self.positions):
            for i_player, player in enumerate(self.players):
                if self.x[(i_player, k_pos)].solution_value() == 1:
                    composition.append((position, player))
        return Team(self.formation, composition)

    def _ban_current_solution(self):
        """Add constraint to avoid same solution"""
        self.solver.Add(
            self.solver.Sum(
                self.x[(i_player, k_pos)]
                for i_player, _ in enumerate(self.players)
                for k_pos, _ in enumerate(self.positions)
                if self.x[(i_player, k_pos)].solution_value() == 1
            )
            <= 10
        )

    def solve(self) -> list[Team]:
        """Find all optimal solutions and return them as teams."""
        solutions: list[Team] = []
        best_obj = -1
        logger.info("Start solving MILP model")
        logger.info(f"formation: {self.formation}")
        logger.info(f"alpha: {self.alpha}")
        while True:
            # solve
            status = self.solver.Solve()
            if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
                # initialize best_obj
                logger.info(
                    f"Solution found! Objective value: {self.solver.Objective().Value()}"
                )
                if not solutions:
                    best_obj = self.solver.Objective().Value()

                # extract team from current solution
                team = self._extract_team_from_solution()

                # if solution is worse than best_obj then break
                if self.solver.Objective().Value() < best_obj:
                    logger.info("Solution found is worse than previous solution. Stop.")
                    break

                # add team to solutions if not already in
                if not any(team.equals(sol) for sol in solutions):
                    solutions.append(team)
                    logger.info("New team added to solutions.")
                else:
                    logger.info("Team already in solutions. Search continues.")

                # add constraint to avoid same solution
                self._ban_current_solution()

            else:
                break
        return solutions


def extract_pareto_frontier(list_teams: list[Team]) -> list[Team]:
    """
    Extract Pareto frontier from a list of teams.

    Parameters
    ----------
    - list_teams: list[Team]
        list of teams

    Returns
    -------
    - list of teams in the Pareto frontier
    """
    pareto_frontier: list[Team] = []
    for team_1 in list_teams:
        if not any(
            team_1.rating <= team_2.rating and team_1.chemistry <= team_2.chemistry
            for team_2 in list_teams
        ):
            pareto_frontier.append(team_1)
    return pareto_frontier


def get_optimal_teams(
    players: Sequence[Player], formation: str | list[str], alpha_step=0.1
) -> list[Team]:
    """
    Get optimal teams for a given formation and a list of players.

    Parameters
    ----------
    - players: Sequence[Player]
        sequence of players
    - formation: str | list[str]
        formation name or list of formation names
    - alpha_step: float, default 0.1
        alpha parameter step to explore Pareto frontier

    Returns
    -------
    - list of optimal teams
    """
    if isinstance(formation, str):
        formation = [formation]
    teams = []
    for form in formation:
        for alpha in np.arange(0, 1 + alpha_step, alpha_step):
            sol = UT_MILP_Model(players, form, alpha).solve()
            teams.extend(sol)
            Team.remove_duplicates(teams)
        model = UT_MILP_Model(players, form)
        teams.extend(model.solve())
    return extract_pareto_frontier(teams)
