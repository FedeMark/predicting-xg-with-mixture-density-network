"""Utils to scrape fbref dataset."""
from typing import Sequence
import pandas as pd
import time


def get_serie_a_data(years: Sequence[int] = range(2017, 2023)):
    data = []
    for year in years:
        print(f"Scraping year {year}")
        time.sleep(2)
        y_data = get_season_data(
            f"https://fbref.com/en/comps/11/{year}-{year+1}/schedule/{year}-{year+1}-Serie-A-Scores-and-Fixtures"
        )
        y_data["starting_year"] = year
        y_data["league"] = "serie_a"
        data.append(y_data)
    data = pd.concat(data)
    data.to_csv(f"serie_a_{year}.csv")

    return data


def get_premier_league_data(years: Sequence[int] = range(2017, 2023)):
    data = []
    for year in years:
        print(f"Scraping year {year}")
        time.sleep(2)
        y_data = get_season_data(
            f"https://fbref.com/en/comps/9/{year}-{year+1}/schedule/{year}-{year+1}-Premier-League-Scores-and-Fixtures"
        )
        y_data["starting_year"] = year
        y_data["league"] = "premier_league"
        data.append(y_data)
    data = pd.concat(data)
    data.to_csv(f"premier_league_{year}.csv")

    return data


def get_bundesliga_data(years: Sequence[int] = range(2017, 2023)):
    data = []
    for year in years:
        print(f"Scraping year {year}")
        time.sleep(2)
        y_data = get_season_data(
            f"https://fbref.com/en/comps/20/{year}-{year+1}/schedule/{year}-{year+1}-Bundesliga-Scores-and-Fixtures"
        )
        y_data["starting_year"] = year
        y_data["league"] = "bundesliga"
        data.append(y_data)
    data = pd.concat(data)
    data.to_csv(f"bundesliga_{year}.csv")

    return data


def get_league_1_data(years: Sequence[int] = range(2017, 2023)):
    data = []
    for year in years:
        print(f"Scraping year {year}")
        time.sleep(2)
        y_data = get_season_data(
            f"https://fbref.com/en/comps/13/{year}-{year+1}/schedule/{year}-{year+1}-Ligue-1-Scores-and-Fixtures"
        )
        y_data["starting_year"] = year
        y_data["league"] = "league_1"
        data.append(y_data)

    data = pd.concat(data)
    data.to_csv(f"league_1_{year}.csv")

    return data


def get_liga_data(years: Sequence[int] = range(2017, 2023)):
    data = []
    for year in years:
        print(f"Scraping year {year}")
        time.sleep(2)
        y_data = get_season_data(
            f"https://fbref.com/en/comps/12/{year}-{year+1}/schedule/{year}-{year+1}-La-Liga-Scores-and-Fixtures"
        )
        y_data["starting_year"] = year
        y_data["league"] = "liga"
        data.append(y_data)

    data = pd.concat(data)
    data.to_csv(f"la_liga{year}.csv")

    return data


def get_primeira_liga_data(years: Sequence[int] = range(2018, 2023)):
    data = []
    for year in years:
        print(f"Scraping year {year}")
        time.sleep(2)
        y_data = get_season_data(
            f"https://fbref.com/en/comps/32/{year}-{year+1}/schedule/{year}-{year+1}-Primeira-Liga-Scores-and-Fixtures"
        )
        y_data["starting_year"] = year
        y_data["league"] = "primeira_liga"
        data.append(y_data)
    data = pd.concat(data)
    data.to_csv(f"primeira_liga_{year}.csv")

    return data


def get_season_data(url: str) -> pd.DataFrame:
    df = pd.read_html(
        url,
        extract_links="body",
    )[0]

    data = []
    for i, row in df.iterrows():
        print(i)
        row_url = row["Match Report"][1]

        if row_url is None:
            continue

        date = row["Date"][0]
        home_team = row["Home"][0]
        away_team = row["Away"][0]

        match_url = "https://fbref.com" + row_url

        try:
            time.sleep(2)
            home_tot = pd.read_html(match_url, header=1)[3].iloc[-1][6:]
            time.sleep(2)
            away_tot = pd.read_html(match_url, header=1)[10].iloc[-1][6:]
            time.sleep(2)

            home_gk_data = pd.read_html(match_url, header=1)[9].iloc[-1][5:]
            time.sleep(2)
            away_gk_data = pd.read_html(match_url, header=1)[16].iloc[-1][5:]

            home_data = pd.concat([home_tot, home_gk_data])
            away_data = pd.concat([away_tot, away_gk_data])

            home_data["Team"] = home_team
            away_data["Team"] = away_team
            home_data["Home"] = True
            away_data["Home"] = False
            home_data["Opponent"] = away_team
            away_data["Opponent"] = home_team
            home_data["Date"] = date
            away_data["Date"] = date

            data.append(home_data)
            data.append(away_data)
        except Exception as e:
            print(e)

    data = pd.concat(data, axis=1).T

    return data
