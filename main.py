# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 08:28:41 2023

@author: Lucas Heller and Felix V. Dehler
"""

# UPGRADED API KEY: 9973533

import base64
import pandas as pd
import requests
import sqlite3
import streamlit as st
import urllib.parse

from collections import OrderedDict
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

THEMEALDB_API_KEY = st.secrets["API_KEY"]
THEMEALDB_API_ENDPOINT = f"https://www.themealdb.com/api/json/v2/{THEMEALDB_API_KEY}"

# DB filepath
DB_FILE = "inventory.db"

# DataFrame pickle containing all the ingredients available on themealdb.com
INGREDIENTS_PICKLE_FILE = "ingredients.pkl"

# Max recipes to display
MAX_RECIPE_PER_PAGE = 10


# Inventory class definition
class Inventory:
    def __init__(self):
        """
        Initialize the new database if the file does not exit
        'inventory' table
            - id: allocated id of the ingredient
            - name: the name of the ingredient
            - expire_date: the expiration date of the ingredient
        """
        self.con = sqlite3.connect(DB_FILE)
        with self.con as DB:
            DB.execute(
                "CREATE TABLE IF NOT EXISTS inventory "
                + "(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, expire_date DATE, ingredient_id INTEGER);"
            )
            DB.execute(
                "CREATE TABLE IF NOT EXISTS meal "
                + "(id INTEGER PRIMARY KEY AUTOINCREMENT, meal_id INTEGER, ingredient_id INTEGER);"
            )

    def add_item(self, name, expire_date, ingredient_id):
        """
        Add a new item and its new meals

        Params:
        - name: str
        - expire_date: datetime

        Return:
        - the new item id (int)
        """
        count = 0
        with self.con as DB:
            new_item_id = DB.execute(
                "INSERT INTO inventory (name, expire_date, ingredient_id) VALUES (?, ?, ?);",
                (name, expire_date, ingredient_id),
            ).lastrowid

            # chekc if the ingredient_id is already fetched
            count = DB.execute(f"SELECT COUNT(*) FROM meal WHERE ingredient_id = {ingredient_id};").fetchone()[0]

        if count == 0:
            filter_api_url = THEMEALDB_API_ENDPOINT + "/filter.php?i="
            try:
                for meal in requests.get(filter_api_url + name).json()["meals"]:
                    meal_id = meal["idMeal"]
                    with self.con as DB:
                        DB.execute(
                            "INSERT INTO meal (meal_id, ingredient_id) VALUES (?, ?);",
                            (meal_id, ingredient_id),
                        )
            except Exception:
                pass
        return new_item_id

    def check_high_priority(self, ingredients):
        """
        Count items and check their priority

        Params:
        - ingredients: list(int) the list of idIngredient from themealdb.com to count

        Return:
        - pd.DataFrame contains the high priority ingredients
        """
        cond = " OR ".join([f"ingredient_id = {x}" for x in ingredients])
        with self.con as DB:
            return pd.read_sql(f"SELECT name FROM inventory WHERE expire_date = DATE('now') AND ({cond});", DB)

    def delete_item(self, item_id):
        """
        Remove a item from the inventory by id

        Params:
        - item_id: the item id to be removed (int)

        Return:
        - None
        """
        with self.con as DB:
            DB.execute(f"DELETE FROM inventory WHERE id = {item_id};")

    def get_items(self):
        """
        Get the available ingredients from the inventory

        Return:
        - df: pd.DataFrame contains the ingredients
        """
        with self.con as DB:
            df = pd.read_sql("SELECT * FROM inventory;", DB)
            df["expire_date"] = df["expire_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
            return df

    def find_meals(self, df_inventory):
        """
        Find meals using any combination of the given ingredient_id

        Params:
        - df: pd.DataFrame conains the inventory with weight

        Return:
        - meals: OrderedDict with key=meal_id, value=tuple(ingredient_id,)
        """
        meals = OrderedDict()

        # calculate weights
        today = date.today()
        df_inventory["weights"] = df_inventory["expire_date"].apply(lambda x: 2 if x <= today else 1 / (x - today).days)

        # first search for high priority ingredients
        ingredients = df_inventory.loc[df_inventory["weights"] == 2, "ingredient_id"].tolist()
        for i in range(len(ingredients), 0, -1):
            for c in combinations(ingredients, i):
                # generate the condition from the combination
                cond = " OR ".join([f"ingredient_id = {x}" for x in c])
                with self.con as DB:
                    df = pd.read_sql(f"SELECT DISTINCT meal_id FROM meal WHERE {cond};", DB)
                    if df.shape[0] == 0:
                        continue
                    for _, meal_id in df["meal_id"].items():
                        if meal_id not in meals.keys():
                            meals[meal_id] = c

        # extend the meals with the low priority ingredients
        ingredients = df_inventory.loc[df_inventory["weights"] != 2, "ingredient_id"].tolist()
        for ing in ingredients:
            with self.con as DB:
                df = pd.read_sql(f"SELECT DISTINCT meal_id FROM meal WHERE ingredient_id = {ing};", DB)
                if df.shape[0] == 0:
                    continue
                for _, meal_id in df["meal_id"].items():
                    if meal_id in meals.keys():
                        meals[meal_id] += (ing,)
                    else:
                        meals[meal_id] = (ing,)

        return meals


# Fetch the recipes at the specific page
def recipe_details(meals, page):
    lookup_api_url = THEMEALDB_API_ENDPOINT + "/lookup.php?i="
    recipes = []
    for i, (meal_id, ingredients) in enumerate(meals.items()):
        if not page * MAX_RECIPE_PER_PAGE < i < (page + 1) * MAX_RECIPE_PER_PAGE:
            continue
        recipe = requests.get(lookup_api_url + str(meal_id)).json()
        recipes.append((recipe["meals"][0], ingredients))
    recipes.sort(key=lambda x: x[1], reverse=True)
    return recipes


###########################################
# Function to download a picture based on a url and return it.
def get_image_as_base64(url):
    response = requests.get(url)
    return base64.b64encode(response.content).decode("utf-8")


# This function downloads all possible ingredtient items, which will be used
# as selection possabilities of the streamlit sidebar to select items
# and then convert the data from download_ingredient_items() into a dataframe
def df_ingredients():
    list_api_url = THEMEALDB_API_ENDPOINT + "/list.php?i=list"
    ingredients_json = requests.get(list_api_url)
    ingredients = ingredients_json.json()

    # Extract the list of meals
    meals_list = ingredients["meals"]
    # Create DataFrame from the list of dictionaries
    df = pd.DataFrame(meals_list)

    return df


def find_recipes(db, df_ingredients):
    # check the current recipe page and save
    page = st.session_state.get("recipe_page", 0)
    st.session_state["recipe_page"] = page

    df = db.get_items()
    meals = db.find_meals(df)

    st.markdown(f"## Found {len(meals)} recipes")

    # search all recipes
    with st.spinner("Fetching the recipes..."):
        recipes = recipe_details(meals, page)

    #checking how many pages of recipes are displayed
    if len(meals) > MAX_RECIPE_PER_PAGE:
        if page == 0:
            if st.button(f"Next {MAX_RECIPE_PER_PAGE} recipes"):
                st.session_state["recipe_page"] += 1
                st.rerun()
        elif (page + 1) * MAX_RECIPE_PER_PAGE < len(meals):
            if st.button(f"Previous {MAX_RECIPE_PER_PAGE} recipes"):
                st.session_state["recipe_page"] -= 1
                st.rerun()
            if st.button(f"Next {MAX_RECIPE_PER_PAGE} recipes"):
                st.session_state["recipe_page"] += 1
                st.rerun()
        else:
            if st.button(f"Previous {MAX_RECIPE_PER_PAGE} recipes"):
                st.session_state["recipe_page"] -= 1
                st.rerun()

    if len(meals) == 0:
        return
    st.markdown(f"#### Showing {page*MAX_RECIPE_PER_PAGE + 1} to {page*MAX_RECIPE_PER_PAGE + 1 + len(recipes)}")

    for recipe in recipes:
        # create an expander for the recipe
        meal_name = recipe[0]["strMeal"]
        high_priority_items = db.check_high_priority(recipe[1])["name"].tolist()
        total = len(recipe[1])
        high = len(high_priority_items)
        expander = st.expander(
            f"{meal_name} (Match {total} ingredients, {high} high priority: {', '.join(high_priority_items)})"
        )
        # print the pic
        expander.image(recipe[0]["strMealThumb"])
        # print all ingredients with measurements

        for key in recipe[0].keys():
            if not key.startswith("strIngredient"):
                continue

            ingredient_key = key
            measure_key = f"strMeasure{key.replace('strIngredient', '')}"
            item = recipe[0][ingredient_key]

            if item == "" or not isinstance(item, str):
                continue

            ingredient_line = f"{item}\t {recipe[0][measure_key]}"
            coop_link = f"https://www.coop.ch/en/search/?text={urllib.parse.quote(item, safe='')}"
            migros_link = f"https://www.migros.ch/en/search?query={urllib.parse.quote(item, safe='')}"

            # check if the recipe contains an inventory item
            if any([item.lower() in x.lower() for x in df["name"].values]):
                expander.write(ingredient_line)
            else:
                ingredient_line = f"""<span style='color:red;'>{ingredient_line}</span>
                    <a href="{coop_link}"><img src="https://www.coop.ch/_ui/23.9.2.533/desktop/common/img/masthead/logo/img/coop_logo.svg" class="shop" /></a>
                    <a href="{migros_link}"><img src="https://www.migros.ch/assets/images/menu/migrosx.svg" class="shop" /></a>
                """
                expander.markdown(ingredient_line, unsafe_allow_html=True)

        # print the instructions
        expander.write(recipe[0]["strInstructions"])
        # put video here
        if recipe[0]["strYoutube"] is not None:
            try:
                expander.video(recipe[0]["strYoutube"])
            except Exception:
                pass


def main():
    # Titel Mainpage via github (only the picture)
    github_image_url = "https://raw.githubusercontent.com/heluc001/SaveThePlate/main/SaveThePlate.jpg"
    base64_image = get_image_as_base64(github_image_url)

    # Create the HTML for the image
    image_tag = f'<img src="data:image/jpeg;base64,{base64_image}" class="logo"/>'


    st.markdown(image_tag, unsafe_allow_html=True)
    # add custom style css
    st.markdown(
        """
       <style>
       .logo {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
            margin-bottom: 20px;
       }
       .shop {
            height: 1em;
       }
       </style>
       """,
        unsafe_allow_html=True,
    )

    # st.title("SaveThePlate")

    # Call df_ingredients to get the DataFrame if not retrieved yet
    ingredients_df = None
    if Path(INGREDIENTS_PICKLE_FILE).exists():
        ingredients_df = pd.read_pickle(INGREDIENTS_PICKLE_FILE)
    else:
        ingredients_df = df_ingredients()
        ingredients_df.to_pickle(INGREDIENTS_PICKLE_FILE)

    # Initialize the database
    db = Inventory()

    # Sidebar
    st.sidebar.title("Food Items and Expiry")

    # Display existing rows
    # Context manager to specify that the following elements should be placed in the app's sidebar
    with st.sidebar:
        # Iterate through the inventory items stored in the database
        for _, item in db.get_items().iterrows():
            # Create three columns in the sidebar with different widths
            col1, col2, col3 = st.columns([3, 2, 1])

            # First Column: Displaying a selection box for choosing a food item
            with col1:
                st.selectbox(
                    f'Food Item ({item["id"]})',  # Display the food item
                    options=ingredients_df["strIngredient"],  # List of options from the 'ingredients_df' DataFrame
                    index=ingredients_df["strIngredient"].tolist().index(item["name"])
                    if item["name"] in ingredients_df["strIngredient"].tolist()
                    else 0,
                    key=f'food_item_{item["id"]}',  # Unique key for the selectbox widget
                )

            # Second Column: Date input for selecting the expire date of the food item
            with col2:
                st.date_input(
                    "Expire Date",  # Label for the date input
                    key=f'expire_date_{item["id"]}',  # Unique key for the date input widget
                    value=item["expire_date"],  # Set the current value of the date input
                    min_value=date.today(),
                )

            # Third Column: Button to delete the current row
            with col3:
                # Red 'X' delete button for each row
                if st.button("❌", key=f'delete_{item["id"]}'):  # Display a red 'X' as a delete button
                    # Call the method with the item_id if the button is clicked
                    db.delete_item(item["id"])
                    # Refresh the page
                    st.session_state["recipe_page"] = 0
                    st.rerun()

        # Create empty item row
        col1, col2, _ = st.columns([3, 2, 1])
        with col1:
            new_item_name = st.selectbox(
                "New Food Item",
                options=ingredients_df["strIngredient"],  # List of options from the 'ingredients_df' DataFrame
            )
        with col2:
            new_expire_date = st.date_input(
                "Expire Date",
                min_value=date.today(),
            )

    # Button to add more rows
    if st.sidebar.button("Add Food Item"):
        try:
            ingredient_id = int(
                ingredients_df[ingredients_df["strIngredient"] == new_item_name].iloc[0]["idIngredient"]
            )
        except Exception:
            st.error(Exception)
        db.add_item(new_item_name, new_expire_date, ingredient_id)
        # Refresh the page
        st.session_state["recipe_page"] = 0
        st.rerun()

    # Save button
    if st.session_state.get("recipe_page", 0) == 0 and st.sidebar.button("Lookup Recipe"):
        # Explanation of st.session_state:
        # session_state in Streamlit is a feature that allows you to preserve and manage
        # the state of user interactions in your Streamlit app.
        # In web applications, "state" refers to the storage and management of data
        # across user interactions. Since Streamlit apps are interactive and dynamic,
        # managing state becomes essential for a seamless user experience.
        st.session_state["recipe_page"] = 0
        find_recipes(db, df_ingredients)
    else:
        find_recipes(db, df_ingredients)


# Run the app
if __name__ == "__main__":
    main()
