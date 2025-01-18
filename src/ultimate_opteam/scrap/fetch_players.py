from typing import Any, Sequence
from selenium import webdriver
from selenium.webdriver.common.by import By

import numpy as np
import time

FUT_URL = "https://www.ea.com/fifa/ultimate-team/web-app/"


def fetch_players(driver: webdriver.Chrome) -> Sequence[dict[str, Any]]:
    driver.get(FUT_URL)
    time.sleep(10)

    # Click on Club
    CLUB_BTN_XPATH = "/html/body/main/section/nav/button[5]"
    driver.find_element(By.XPATH, CLUB_BTN_XPATH).click()
    time.sleep(0.5)

    # Get number of players in club
    NB_PLAYERS_TEXT_XPATH = (
        "/html/body/main/section/section/div[2]/div/div/div[1]/div[2]/div/span"
    )
    nb_players_text = driver.find_element(By.XPATH, NB_PLAYERS_TEXT_XPATH).text
    nb_players = int(nb_players_text.split(" ")[0])
    print("nb. players in club:", nb_players)

    # Click on Players tile
    PLAYERS_TILE_XPATH = "/html/body/main/section/section/div[2]/div/div/div[1]"
    driver.find_element(By.XPATH, PLAYERS_TILE_XPATH).click()
    time.sleep(0.5)

    # Loop over all players to get info
    MAX_PLAYERS_PER_PAGE = 20
    nb_pages = int(np.ceil(nb_players / MAX_PLAYERS_PER_PAGE))
    players_list = []
    relevant_info_field = [
        "Full Name",
        "Preferred Position",
        "Alternate positions",
        "Nation",
        "Club",
        "League",
    ]
    for page in range(nb_pages):
        nb_players_on_page = (
            nb_players % MAX_PLAYERS_PER_PAGE
            if page == nb_pages - 1
            else MAX_PLAYERS_PER_PAGE
        )
        PLAYER_LIST_PAGE_XPATH = [
            f"/html/body/main/section/section/div[2]/div/div/div/div[3]/ul/li[{i_player}]"
            for i_player in range(1, nb_players_on_page + 1)
        ]
        for xpath in PLAYER_LIST_PAGE_XPATH:
            player_info: dict[str, Any] = {}

            # Click on player
            driver.find_element(By.XPATH, xpath).click()
            time.sleep(0.2)

            # Click on Player Bio
            PLAYER_BIO_BTN_XPATH = "/html/body/main/section/section/div[2]/div/div/section/div/div/div[2]/div[2]/button[1]"
            driver.find_element(By.XPATH, PLAYER_BIO_BTN_XPATH).click()
            time.sleep(0.2)

            # Loop over all bio info
            info_list = driver.find_elements(By.CLASS_NAME, "ut-item-bio-row-view")
            for info in info_list:
                info_text = info.text.split("\n")
                if info_text[0] in relevant_info_field:
                    player_info[info_text[0]] = info_text[1]

            # Click on attribute
            ATTRIBUTE_BTN_XPATH = "/html/body/main/section/section/div[2]/div/div/section/div[2]/article/div[2]/div/button[2]"
            driver.find_element(By.XPATH, ATTRIBUTE_BTN_XPATH).click()
            time.sleep(0.2)

            # Get overall rating
            OVERALL_RATING_TEXT_XPATH = "/html/body/main/section/section/div[2]/div/div/section/div[2]/article/div[3]/div/ul[1]/li[1]/span"
            player_info["Overall Rating"] = int(
                driver.find_element(By.XPATH, OVERALL_RATING_TEXT_XPATH).text
            )

            players_list.append(player_info)

        # Click on Next page
        if page < nb_pages - 1:
            NEXT_BTN_XPATH = "/html/body/main/section/section/div[2]/div/div/div/div[3]/div/button[2]"
            driver.find_element(By.XPATH, NEXT_BTN_XPATH).click()
            time.sleep(0.2)

    return players_list
