from typing import List, Sequence

from ultimate_opteam.data import Player


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
    def chemistry(self) -> int:
        """Team chemistry."""
        category_score = {
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
                for cat in ["league", "nation"]:
                    chem += _get_mode(category_score[cat][getattr(player, cat)])
                chem += _get_mode(category_score["club"][player.club], [2, 5, 8])
                player_chem.append((pos, chem))
        return player_chem
