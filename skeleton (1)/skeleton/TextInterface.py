


class TextInterface:

    def __init__(self):
        self.current_step = 0  # Track the current step in the interaction process.

    @staticmethod
    def display_options():

        print("\n--- Email Classification System ---")
        print("1. Load and preprocess data")
        print("2. Generate embeddings")
        print("3. Train a model")
        print("4. Predict email types")
        print("5. Evaluate the model")
        print("6. Exit")

    def handle_choice(choice, main_class):

        if choice == 1:
            print("Loading and preprocessing data...")
            main_class.load_and_preprocess_data()
        elif choice == 2:
            print("Generating embeddings...")
            main_class.generate_embeddings()
        elif choice == 3:
            print("Training the model...")
            main_class.train_model()
        elif choice == 4:
            print("Predicting email types...")
            main_class.predict()
        elif choice == 5:
            print("Evaluating the model...")
            main_class.evaluate_model()
        elif choice == 6:
            print("Exiting the system. Goodbye!")
            exit()
        else:
            print("Invalid choice. Please try again.")
    def get_user_choice(self):

        try:
            choice = int(input("Enter your choice: "))
            return choice
        except ValueError:
            print("Invalid input. Please enter a number.")
            return self.get_user_choice()
