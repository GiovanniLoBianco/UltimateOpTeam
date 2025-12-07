from unittest.mock import patch, mock_open
import pandas as pd
from ultimate_opteam.data.player import Player, get_players_from_csv


def test_player_initialization(sample_player_1: Player):
    assert sample_player_1.name == "Lionel Messi"
    assert sample_player_1.rating == 93
    assert not sample_player_1.icon
    assert not sample_player_1.hero


def test_can_play_at(sample_player_1: Player):
    assert sample_player_1.can_play_at("RW")
    assert sample_player_1.can_play_at("ST")
    assert sample_player_1.can_play_at("CF")
    assert not sample_player_1.can_play_at("GK")


@patch("ultimate_opteam.data.player.pd.read_csv")
@patch("ultimate_opteam.data.player.json.load")
@patch("builtins.open", new_callable=mock_open)
def test_get_players_from_csv(mock_file, mock_json, mock_read_csv):
    # Mock DataFrame
    data = {
        "Full Name": ["Player 1"],
        "Preferred Position": ["ST"],
        "Alternate positions": ["CF,LW"],
        "Overall Rating": [85],
        "Nation": ["Nation 1"],
        "League": ["League 1"],
        "Club": ["Club 1"],
    }
    df = pd.DataFrame(data)
    mock_read_csv.return_value = df

    # Mock JSON league pairs
    mock_json.return_value = {"League 1": "Mapped League 1"}

    players = get_players_from_csv("dummy_path.csv")

    assert len(players) == 1
    p = players[0]
    assert p.name == "Player 1"
    assert p.preferred_position == "ST"
    assert "CF" in p.alternate_position
    assert "LW" in p.alternate_position
    assert p.league == "Mapped League 1"  # Check mapping


def test_can_play_at_edge_cases(sample_player_1: Player):
    # Test with empty alternate positions
    sample_player_1.alternate_position = []
    assert sample_player_1.can_play_at("RW")
    assert not sample_player_1.can_play_at("ST")


@patch("ultimate_opteam.data.player.pd.read_csv")
@patch("ultimate_opteam.data.player.json.load")
@patch("builtins.open", new_callable=mock_open)
def test_get_players_from_csv_empty(mock_file, mock_json, mock_read_csv):
    # Mock Empty DataFrame
    df = pd.DataFrame(
        columns=[
            "Full Name",
            "Preferred Position",
            "Alternate positions",
            "Overall Rating",
            "Nation",
            "League",
            "Club",
        ]
    )
    mock_read_csv.return_value = df
    mock_json.return_value = {}

    players = get_players_from_csv("dummy_path.csv")
    assert len(players) == 0
