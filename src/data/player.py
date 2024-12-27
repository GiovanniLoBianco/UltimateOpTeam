from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass
class Player:
    """
    Player dataclass.

    Attributes
    ----------
    - name: player name
    - preferred_position: player position
    - alternate_position: player alternative position
    - rating: player overall rating
    - nation: player nation
    - league: player league
    - club: player club
    - icon: True iff player is an icon
    - hero: True iff player is a hero
    """

    name: str
    preferred_position: str
    alternate_position: Sequence[str]
    rating: int
    nation: str
    league: str
    club: str
    icon: bool = False
    hero: bool = False

    def can_play_at(self, position: str) -> bool:
        """
        Check if the player can play at a given position.

        Parameters
        ----------
        - position: position to check

        Returns
        -------
        - boolean indicating if the player can play at the given position
        """
        return (
            position == self.preferred_position or position in self.alternate_position
        )


def get_players_from_csv(path_csv: Path) -> Sequence[Player]:
    """
    Read the players from a csv file.

    Parameters
    ----------
    - path_csv: path to the csv file

    Returns
    -------
    - list of players
    """
    players = []
    df_players = pd.read_csv(path_csv, dtype={"rating": int})
    for _, row in df_players.iterrows():
        players.append(
            Player(
                name=row["Full Name"],
                preferred_position=row["Preffered Position"],
                alternate_position=row["Alternate positions"].split(","),
                rating=row["Overall Rating"],
                nation=row["Nation"],
                league=row["League"],
                club=row["Club"],
            )
        )
    return players
