# This is a main file: The controller. All methods will directly or indirectly be called here

from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
from Config import Config
from TextInterface import *
from MainController import *
from preprocess import preprocess_data, de_duplication, noise_remover
import os

seed = 0
random.seed(seed)
np.random.seed(seed)


def display_main_menu():
    print("\n--- Email Classification System ---")
    print("1. AppGallery")
    print("2. Purchasing")
    print("3. Exit")


def load_dataset(option):
    base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    if option == 1:
        filepath = os.path.join(base_path, "data", "AppGallery.csv")
    elif option == 2:
        filepath = os.path.join(base_path, "data", "Purchasing.csv")
    else:
        return None

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    return filepath


if __name__ == '__main__':
    while True:
        display_main_menu()
        try:
            dataset_choice = int(input("Enter your choice (1-3): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 3.")
            continue

        if dataset_choice == 3:
            print("Exiting the system. Goodbye!")
            break

        filepath = load_dataset(dataset_choice)
        if not filepath:
            print("Invalid choice or file not found. Please try again.")
            continue

        # Preprocess the selected dataset
        df = preprocess_data(filepath)
        print("Dataset preprocessed successfully!")

        # Generate embeddings
        embeddings, df_with_embeddings = get_embeddings(df)
        print("Embeddings generated successfully!")

        # Display categories
        category_map = {
            "Functionality Issues": "y2",
            "Refund Requests": "y3",
            "Feedback": "y1",
            "Other": None  # Represents uncategorized emails
        }
        print("\n--- Select a Category to Display Emails ---")
        for idx, category_name in enumerate(category_map.keys(), start=1):
            print(f"{idx}. {category_name}")

        # Process user selection
        try:
            category_choice = int(input("Enter your choice: "))
            if category_choice < 1 or category_choice > len(category_map):
                raise ValueError("Invalid choice.")
        except ValueError:
            print("Invalid input. Returning to the main menu...")
            continue

        chosen_category = list(category_map.keys())[category_choice - 1]
        category_column = category_map[chosen_category]

        # Filter and categorize emails
        if category_column:
            categorized_emails = train_and_categorize(embeddings, df_with_embeddings, category_column)
        else:
            # Handle "Other" category
            categorized_emails = df_with_embeddings[
                ~df_with_embeddings["y"].isin(category_map.values())
            ]["Interaction content"].tolist()


        # Display results
        if categorized_emails:
            print(f"\n--- Emails for Category: {chosen_category} ---")
            for idx, email in enumerate(categorized_emails, start=1):
                print(f"{idx}. {email}\n")  # Print each email on a new line with an index
        else:
            print(f"No valid emails found for the selected category: {chosen_category}.")

        input("\nPress Enter to return to the main menu...")


