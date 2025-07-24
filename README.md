## Introduction
Ultimate OpTeam is a project designed to optimize your team in EA FC's Ultimate Team online
multiplayer mode. Given the set of players from your Club, the objective is to generate the Pareto
Frontier of the Ultimate Team Optimization Problem (UTOP), which attempts to find the best
combination of players that would maximize both the general score and the team's chemistry.

## Features
- Tools for scraping your Club from the Ultimate Team Web App, in order to pull all your players'
  information.
- MILP model for finding the optimal Pareto frontier.

# Docs
- MILP model for UTOP and iterative search to find the pareto front.

## Next steps
- Create a Constraint Programming model and benchmark Google OR-Tools and Choco solver on it.
- Compare the Constraint Programming approach to the MILP model.
- Create a Dash app integrating the solver, the scraping plugin and some data visualization.
- Add installation guide to README.md

## Developer's Note
This is a side project I am doing on my spare time. I am mostly interested in the scientific
challenge of the UTOP, as well as practicing my coding skills and exploring new python libraries. I
am not entirely certain that this project would be very helpful to seasoned Ultimate Team players,
as there are many aspects of the game that are not considered and many subjective choices. That
being said, this solver is saving me a lot of time when I attempt to optimze my own team. I will
update the README.md at each new added features.

## Disclaimer
Please note that this project involves scraping data from the Ultimate Team Web App to gather the
necessary information about players from user's Club. This is currently the only method available to
pull all the required data for the optimization process. The scraping process is designed to be very
light and efficient, requiring updates only once every few hours, days, or weeks to refresh the
current club information. The number of requests made to the server during scraping is minimal and
does not cause any significant load.
