from typing import Sequence

from ultimate_opteam.data import Player
import scipy.optimize as sco
import numpy as np


class Team:
    """Class defining a team."""

    def __init__(self, formation: str, composition: Sequence[tuple[str, Player]]):
        """
        Attributes
        ---------
        formation: str
            formation name.
        composition: Sequence[tuple[str, Player]]
            sequence of tuples containing player position and player.
            Example: [("GK", player1), ("CB", player2), ...]
        """
        self.formation = formation
        self.composition = composition

    @property
    def rating(self) -> int:
        """Team rating."""
        return sum(player.rating for _, player in self.composition)

    @property
    def chemistry(self) -> list[tuple[str, int]]:
        """Team chemistry."""
        category_score: dict = {
            "club": {},
            "league": {},
            "nation": {},
        }
        for _, player in self.composition:
            for cat in category_score.keys():
                if getattr(player, cat) not in category_score[cat]:
                    category_score[cat][getattr(player, cat)] = 1
                else:
                    category_score[cat][getattr(player, cat)] += 1
                if player.icon and cat == "nation":
                    category_score[cat][player.nation] += 1
                if player.hero and cat == "league":
                    category_score[cat][player.league] += 1

        def _get_mode(score: int, thr: list[int] = [3, 5, 8]):
            if score >= thr[2]:
                return 3
            elif score >= thr[1]:
                return 2
            elif score >= thr[0]:
                return 1
            return 0

        player_chem = []
        for pos, player in self.composition:
            if player.icon or player.hero:
                player_chem.append((pos, 3))
            else:
                chem = 0
                chem += _get_mode(category_score["league"][player.league], [2, 4, 7])
                chem += _get_mode(category_score["nation"][player.nation], [2, 5, 8])
                chem += _get_mode(category_score["club"][player.club], [2, 4, 7])
                chem = min(3, chem)
                player_chem.append((pos, chem))
        return player_chem

    def equals(self, other: "Team") -> bool:
        """Check if two teams have the same formation and the same players, no matter where they are
        positioned."""
        if self.formation != other.formation:
            return False
        for _, player in enumerate(self.composition):
            if player not in other.composition:
                return False
        return True

    @staticmethod
    def remove_duplicates(teams: list["Team"]) -> list["Team"]:
        """Remove duplicate teams from a sequence of teams."""
        _teams = teams.copy()
        filtered = []
        while len(_teams) > 0:
            team = _teams.pop(0)
            filtered.append(team)
            _teams = [rem_team for rem_team in _teams if not rem_team.equals(team)]
        return filtered

    @property
    def players(self) -> list[Player]:
        """List of players in the team."""
        return [player for _, player in self.composition]

    def optimize(self) -> "Team":
        """Reassign players to positions using the Hungarian algorithm to maximize player's
        preferrences."""

        def calculate_gain(player: Player, position: str):
            if position == player.preferred_position:
                return 2
            elif position in player.alternate_position:
                return 1
            return -30

        positions = [pos for pos, _ in self.composition]
        players = [player for _, player in self.composition]

        cost_matrix = np.zeros((len(players), len(positions)))

        for i, player in enumerate(players):
            for j, position in enumerate(positions):
                cost_matrix[i, j] = calculate_gain(player, position)

        row_ind, col_ind = sco.linear_sum_assignment(cost_matrix, maximize=True)

        optimized_composition = [
            (positions[j], players[i]) for i, j in zip(row_ind, col_ind)
        ]

        return Team(self.formation, optimized_composition)
