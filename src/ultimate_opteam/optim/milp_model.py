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

logger = loguru.logger


class UT_MILP_Model:
    """
    MILP model for the Ultimate Team problem.
    """

    def __init__(
        self,
        players: Sequence[Player],
        formation: str,
        alpha: float = 0.5,
        pareto_frontier: list[Team] | None = None,
    ):
        """
        Attributes
        ----------
        - players: Sequence[Player]
            sequence of players
        - formation: str
            formation name
        - alpha: float, default 0.5
            The weight of team chemistry in the objective function. The weight of team rating is 1 -
            alpha.
        - pareto_frontier: list[Team] | None, optional (default None)
            list of teams on the pareto frontier.
        """
        self.players = tuple(players)
        self.formation = formation
        self.alpha = alpha
        self.solver: pywraplp.Solver = pywraplp.Solver.CreateSolver("SCIP")
        self.pareto_frontier = pareto_frontier

        # Variables
        self.x: dict = {}
        self.y: dict = {}
        self.gamma: dict = {}
        self.chemistry: dict = {}
        self.final_chemistry: dict = {}
        self.objective_var = {}
        if pareto_frontier is not None:
            self.pareto_frontier_var: dict = {}
        self._declare_variables()

        # Constraints
        self.constraints: list = []
        self._add_constraint()

        # Objective
        self._add_objective()

    @cached_property
    def positions(self) -> tuple[str, ...]:
        path_formation = Path(
            os.environ["ULTIMATE_OPTEAM_PROJECT_PATH"]
            + "/src/ultimate_opteam/data/formations.json"
        )
        with open(path_formation, "r") as file:
            formations = json.load(file)
            return formations[self.formation]

    @cached_property
    def nation(self) -> tuple[str, ...]:
        return tuple(set([player.nation for player in self.players]))

    @cached_property
    def league(self) -> tuple[str, ...]:
        return tuple(set([player.league for player in self.players]))

    @cached_property
    def club(self) -> tuple[str, ...]:
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

        # pareto frontier variable
        if self.pareto_frontier is not None:
            for i_team, team in enumerate(self.pareto_frontier):
                self.pareto_frontier_var[f"team_{i_team}"] = {
                    "above_rating": self.solver.BoolVar(f"is_above_rating_{i_team}"),
                    "above_chemistry": self.solver.BoolVar(
                        f"is_above_chemistry_{i_team}"
                    ),
                    "rating_ratio": self.solver.NumVar(
                        lb=0, name=f"rating_ratio_{i_team}"
                    ),
                    "chemistry_ratio": self.solver.NumVar(
                        lb=0, name=f"chemistry_ratio_{i_team}"
                    ),
                }

        # objective vars
        self.objective_var["rating"] = self.solver.NumVar(
            lb=0.0, ub=1.0, name="obj_rating"
        )
        self.objective_var["chemistry"] = self.solver.NumVar(
            lb=0.0, ub=1.0, name="obj_chemistry"
        )

    def _add_constraint(self):
        self._add_players_assignment_constraint()
        self._add_category_coherence_constraint()
        self._add_score_mode_constraint()
        self._add_score_position_constraint()
        if self.pareto_frontier is not None:
            self._ban_team_from_pareto_frontier()
            self._add_pareto_frontier_constraint()

    def _add_players_assignment_constraint(self):
        """
        Constraints related to players to positions assignment.
        """
        for k_pos, _ in enumerate(self.positions):
            self.solver.Add(
                self.solver.Sum(
                    self.x[(i_player, k_pos)] for i_player, _ in enumerate(self.players)
                )
                == 1,
                f"position_{k_pos}_must_be_filled",
            )

        for i_player, _ in enumerate(self.players):
            self.solver.Add(
                self.solver.Sum(
                    self.x[(i_player, k_pos)] for k_pos, _ in enumerate(self.positions)
                )
                <= 1,
                f"player_{i_player}_can_only_play_at_one_position",
            )

        for i_player, player in enumerate(self.players):
            for k_pos, position in enumerate(self.positions):
                if not player.can_play_at(position):
                    self.solver.Add(
                        self.x[(i_player, k_pos)] == 0,
                        f"player_{i_player}_cannot_play_at_position_{k_pos}",
                    )

    def _add_category_coherence_constraint(self):
        """
        Constraints related to category to position assignment.
        """
        for cat in ["nation", "league", "club"]:
            for k_pos, _ in enumerate(self.positions):
                for j_cat, name_cat in enumerate(getattr(self, cat)):
                    self.solver.Add(
                        self.solver.Sum(
                            self.x[(i_player, k_pos)]
                            for i_player, player in enumerate(self.players)
                            if getattr(player, cat) == getattr(self, cat)[j_cat]
                        )
                        >= self.y[cat][(k_pos, j_cat)],
                        f"position_{k_pos}_has_same_{cat}_{name_cat}_as_assigned_player",
                    )

        for k_pos, _ in enumerate(self.positions):
            for j_nat, name_nat in enumerate(self.nation):
                self.solver.Add(
                    self.solver.Sum(
                        self.x[(i_player, k_pos)]
                        for i_player, player in enumerate(self.players)
                        if player.nation == self.nation[j_nat] and player.icon
                    )
                    >= self.y["icon"][(k_pos, j_nat)],
                    f"position_{k_pos}_is_icon_of_{name_cat}_if_assigned_player_is",
                )

        for k_pos, _ in enumerate(self.positions):
            for j_league, name_league in enumerate(self.league):
                self.solver.Add(
                    self.solver.Sum(
                        self.x[(i_player, k_pos)]
                        for i_player, player in enumerate(self.players)
                        if player.league == self.league[j_league] and player.hero
                    )
                    >= self.y["hero"][(k_pos, j_league)],
                    f"position_{k_pos}_is_hero_of_{name_league}_if_assigned_player_is",
                )

    def _add_score_mode_constraint(self):
        """
        Constraints related to how score is computed for each nation, club and league.
        """

        for j_nation, name_nation in enumerate(self.nation):
            self.solver.Add(
                2 * self.gamma["nation"][(j_nation, 1)]
                + 5 * self.gamma["nation"][(j_nation, 2)]
                + 8 * self.gamma["nation"][(j_nation, 3)]
                <= self.solver.Sum(
                    self.y["nation"][(k_pos, j_nation)]
                    for k_pos, _ in enumerate(self.positions)
                )
                + self.solver.Sum(
                    self.y["icon"][(k_pos, j_nation)]
                    for k_pos, _ in enumerate(self.positions)
                ),
                f"nation_{name_nation}_score_mode_constraint",
            )
            self.solver.Add(
                self.solver.Sum(
                    self.gamma["nation"][(j_nation, mode)] for mode in range(4)
                )
                <= 1,
                f"only_one_score_mode_for_nation_{name_nation}",
            )

        for j_league, name_league in enumerate(self.league):
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
                ),
                f"league_{name_league}_score_mode_constraint",
            )
            self.solver.Add(
                self.solver.Sum(
                    self.gamma["league"][(j_league, mode)] for mode in range(4)
                )
                <= 1,
                f"only_one_score_mode_for_league_{name_league}",
            )

        for j_club, name_club in enumerate(self.club):
            self.solver.Add(
                2 * self.gamma["club"][(j_club, 1)]
                + 4 * self.gamma["club"][(j_club, 2)]
                + 7 * self.gamma["club"][(j_club, 3)]
                <= self.solver.Sum(
                    self.y["club"][(k_pos, j_club)]
                    for k_pos, _ in enumerate(self.positions)
                ),
                f"club_{name_club}_score_mode_constraint",
            )
            self.solver.Add(
                self.solver.Sum(self.gamma["club"][(j_club, mode)] for mode in range(4))
                <= 1,
                f"only_one_score_mode_for_club_{name_club}",
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
                ),
                f"player_{i_player}_chemistry_constraint",
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
                    + self.chemistry["player"][i_player],
                    f"position_{k_pos}_chemistry_is_inferior_to_player_{i_player}_chemistry",
                )

            self.solver.Add(
                self.final_chemistry[k_pos]
                <= M
                * self.solver.Sum(
                    self.x[(i_player, k_pos)]
                    for i_player, player in enumerate(self.players)
                    if player.icon or player.hero
                )
                + self.chemistry["position"][k_pos],
                f"position_{k_pos}_final_chemistry_constraint",
            )

    def _add_objective(self):
        self.solver.Add(
            self.objective_var["chemistry"]
            <= self.solver.Sum(
                1 / 33 * self.final_chemistry[k_pos]
                for k_pos, _ in enumerate(self.positions)
            ),
            "obj_chemistry_def",
        )
        self.solver.Add(
            self.objective_var["rating"]
            <= self.solver.Sum(
                player.rating / 1100 * self.x[(i_player, k_pos)]
                for i_player, player in enumerate(self.players)
                for k_pos, _ in enumerate(self.positions)
            ),
            "obj_rating_def",
        )

        self.solver.Maximize(
            self.alpha * self.objective_var["chemistry"]
            + (1 - self.alpha) * self.objective_var["rating"]
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

    def _ban_team_from_pareto_frontier(self):
        """Add constraint to avoid solution with same players."""
        for i_team, team in enumerate(self.pareto_frontier):
            self.solver.Add(
                self.solver.Sum(
                    self.x[(i_player, k_pos)]
                    for i_player, player in enumerate(self.players)
                    for k_pos, _ in enumerate(self.positions)
                    if player in team.players
                )
                <= 10,
                f"ban_team_{i_team}",
            )

    def _add_pareto_frontier_constraint(self):
        for i_team, team in enumerate(self.pareto_frontier):
            chemistry = team.chemistry / 33.0
            rating = team.rating / 100
            self.solver.Add(
                self.pareto_frontier_var[f"team_{i_team}"]["above_rating"]
                + self.pareto_frontier_var[f"team_{i_team}"]["above_chemistry"]
                + 0.5 * self.pareto_frontier_var[f"team_{i_team}"]["rating_ratio"]
                + 0.5 * self.pareto_frontier_var[f"team_{i_team}"]["chemistry_ratio"]
                >= 1,
                f"pareto_constraint_{i_team}",
            )
            self.solver.Add(
                self.objective_var["chemistry"]
                > chemistry
                + self.pareto_frontier_var[f"team_{i_team}"]["above_chemistry"]
                - 1,
                f"pareto_above_chemistry_{i_team}",
            )
            self.solver.Add(
                self.objective_var["rating"]
                > rating
                + self.pareto_frontier_var[f"team_{i_team}"]["above_rating"]
                - 1,
                f"pareto_above_rating_{i_team}",
            )
            self.solver.Add(
                self.pareto_frontier_var[f"team_{i_team}"]["rating_ratio"]
                <= self.objective_var["rating"] / rating,
                f"pareto_rating_ratio_{i_team}",
            )
            self.solver.Add(
                self.pareto_frontier_var[f"team_{i_team}"]["chemistry_ratio"]
                <= self.objective_var["chemistry"] / rating,
                f"pareto_rating_chemistry_{i_team}",
            )

    def _get_current_objective(self):
        """Return current objective value, rounded to 6 decimals, to avoid float errors"""
        return np.round(self.solver.Objective().Value(), decimals=6)

    def solve(self) -> Team | None:
        """Find all optimal solutions and return them as teams."""
        logger.info("Start solving MILP model")
        logger.info(f"formation: {self.formation}")
        logger.info(f"alpha: {self.alpha}")
        # solve
        status = self.solver.Solve()
        if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            # initialize best_obj
            logger.info(
                f"Solution found! Objective value: {self._get_current_objective()}"
            )

            # extract team from current solution
            team = self._extract_team_from_solution()

            # optimize team (reassign players to positions based on position preferences)
            team = team.optimize()

            return team

        else:
            logger.info("No solution found!")
            return None


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
            (team_1.rating <= team_2.rating and team_1.chemistry < team_2.chemistry)
            or (team_1.rating < team_2.rating and team_1.chemistry <= team_2.chemistry)
            for team_2 in list_teams
            if not team_1.equals(team_2)
        ):
            pareto_frontier.append(team_1)
    return pareto_frontier


def get_optimal_teams(
    players: Sequence[Player], formation: str | list[str], alpha_step=0.05
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
    teams: list[Team] = []
    for form in formation:
        logger.info(f"Search solutions for formation: {form}")
        while True:
            sol = UT_MILP_Model(
                players,
                form,
                alpha=0.5,
                pareto_frontier=teams if len(teams) > 0 else None,
            ).solve()
            if sol is not None:
                teams.append(sol)
                teams = extract_pareto_frontier(teams)
            else:
                break
    return teams
