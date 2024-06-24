"""
This file is intended to contain the code for the GUI of Dishcovery,
and will call methods from graph.py to display the results.

Authors: Yiping Chen, Alan Su, Lily Phan, Defne Eris
Professor: Sadia Sharmin
Course: CSC111, Introduction to Computer Science
Date: April 2024
"""

# Import statements
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as st
from graph import WeightedGraph
from graph import load_weighted_review_graph


class DishcoveryGUI(tk.Tk):
    """
    This is the main controller for the Dishcovery GUI. The 2 frames used in the GUI will be controlled
    by this class, in that this class will be calling those frames and when they should appear.

    This class also defines the initializations of the GUI, such as the GUI size and title.

    Instance Attributes:
        - frames: Contains a mapping between each frame and the screen it will show.
    """
    frames: dict

    def __init__(self, *args, **kwargs) -> None:
        # Setting up the GUI
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Dishcovery')
        self.geometry("1280x790")

        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        # Setting up the frames
        self.frames = {Screen1: Screen1(frame, self), Screen2: Screen2(frame)}
        self.frames[Screen1].grid(row=0, column=0, sticky="nsew")
        self.frames[Screen2].grid(row=0, column=0, sticky="nsew")

        self.show_frame(Screen1)

    def show_frame(self, cont: tk.Frame) -> None:
        """
        Depending on which frame gets passed in as cont, the frame will appear on the GUI.

        Preconditions:
            - cont in self.frames.keys()
        """
        curr_frame = self.frames[cont]
        curr_frame.tkraise()


class Screen1(tk.Frame):
    """
    Starting splashscreen for the GUI.

    Instance Attributes:
        - background_image: The background image for the GUI.
        - background_label: Used to help place background_image in place.
    """
    background_image: tk.PhotoImage
    background_label: tk.Label

    # Setting up the frame to put the image on
    def __init__(self, parent: tk.Frame, controller: DishcoveryGUI) -> None:
        tk.Frame.__init__(self, parent)

        # Defining the image
        self.background_image = tk.PhotoImage(file="splashscreen.png")
        self.background_label = tk.Label(self, image=self.background_image)
        self.background_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Adding the start button
        style = ttk.Style()
        style.configure('S.TButton', font=('Helvetica', 20, 'bold'), foreground='#FFB0CA', border='#C2D9DF')
        start_button = ttk.Button(self, text="Click Here To Begin!", style='S.TButton',
                                  command=lambda: controller.show_frame(Screen2))
        start_button.grid(row=1, column=1, padx=500, pady=374)


class Screen2(tk.Frame):
    """ th
    Displays in the GUI the main screen where the user will enter their criteria for a recipe and be shown
    their recommendations.

    Instance Attributes:
        - background_image: The background image for the GUI.
        - background_label: Used to help place background_image in place.
        - food_graph: Contains the graph with all the recipe and user information.
        - criteria: Used to hold the criteria the user wants to filter their recipe by, including
                    which recipe they chose, recommendation system preference, number of recommendations,
                    and whether they want to incorporate the sentiment analysis as well.
    """
    background_image: tk.PhotoImage
    background_label: tk.Label
    food_graph: WeightedGraph
    criteria: dict

    def __init__(self, parent: tk.Frame) -> None:
        tk.Frame.__init__(self, parent)

        # Defining the image
        self.background_image = tk.PhotoImage(file="mainscreen.png")
        self.background_label = tk.Label(self, image=self.background_image)
        self.background_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.food_graph = load_weighted_review_graph()
        # Setting the criteria to the default
        self.criteria = {'Chosen Recipe': '24k carrots', 'Chosen Rec System': 'pagerank',
                         'Num Recs': 0, 'Sentimental': False}

        # Adding in the text at the top of the screen
        title_text = tk.Label(self, text="Welcome to Dishcovery!",
                              font=("Helvetica", 30, "bold"), fg='#8f2246', bg="white")
        title_text.place(relx=0.25, rely=0.1, relwidth=0.5, relheight=0.1)

        explanation_text = tk.Label(self,
                                    text="""Look through the recipes below and select the one you like best.
        Also choose which recommendation system you'd like to use, whether you'd like to use sentiment score,
        and the number of recommendations you want. Press confirm when you find your recipe and done
        when you're ready to continue.""",
                                    font=("Helvetica", 15), fg='#c9557b', bg="white")
        explanation_text.place(relx=0.1, rely=0.18, relwidth=0.8, relheight=0.15)

        self.display_choices()

    def display_choices(self) -> None:
        """
        This method will display all the choices the user needs to make in determining their criteria for
        a recipe type.
        The user will need to choose their desired recipe, their preferred recommendation system, number of
        recommendations and also whether they want their recommendations to be organized by sentiment score.
        """

        # Displays the scrolling menu to choose the recipe by name
        recipe_text = tk.Label(self, text="Choose the Recipe:",
                               font=("Helvetica", 12, 'bold'), fg='#8f2246', bg="white")
        recipe_text.place(relx=0.09, rely=0.28, relwidth=0.15, relheight=0.1)
        recipes = tk.Listbox(self, selectmode="single", font=('Helvetica', 15))
        for item in list(self.food_graph.get_all_recipe_names()):
            recipes.insert(tk.END, item.upper())
        recipes.place(relx=0.1, rely=0.35, relwidth=0.37, relheight=0.44)

        # Displays the dropdown menu to choose the preferred recommendation system
        rec_text = tk.Label(self, text="Choose the Recommendation System:",
                            font=("Helvetica", 12, 'bold'), fg='#8f2246', bg="white")
        rec_text.place(relx=0.5, rely=0.4, relwidth=0.4, relheight=0.07)
        system = tk.StringVar()
        dropdown = tk.OptionMenu(self, system,
                                 *['JACCARD', 'OVERLAP', 'COSINE',
                                   'TANIMOTO', 'SIMRANK', 'SIMRANK_PPLUS', 'PAGERANK'])
        dropdown.place(relx=0.5, rely=0.47, relwidth=0.4, relheight=0.03)

        # Displays an entry box to choose the number of recommendations the user wants
        num_text = tk.Label(self, text="Choose the Number of Recommendations:",
                            font=("Helvetica", 12, 'bold'), fg='#8f2246', bg="white")
        num_text.place(relx=0.5, rely=0.5, relwidth=0.4, relheight=0.07)
        num_rec = tk.Entry(self, textvariable=tk.IntVar())
        num_rec.place(relx=0.6, rely=0.55, relwidth=0.2, relheight=0.05)

        # Displays a dropdown box for whether the user wants to use the sentiment score or not
        sen_text = tk.Label(self, text="Sentiment:",
                            font=("Helvetica", 12, 'bold'), fg='#8f2246', bg="white")
        sen_text.place(relx=0.5, rely=0.6, relwidth=0.4, relheight=0.07)
        choice = tk.StringVar()
        dropdown1 = tk.OptionMenu(self, choice, *['YES', 'NO'])
        dropdown1.place(relx=0.5, rely=0.65, relwidth=0.4, relheight=0.03)

        # Adding the confirm button
        style = ttk.Style()
        style.configure('A.TButton', font=('Helvetica', 17, 'bold'), foreground='#ffd6e4', border='#C2D9DF')
        confirm_button = ttk.Button(self, text="Confirm Choice", style='A.TButton',
                                    command=lambda: self.assign_val(recipes, system, num_rec, choice))
        confirm_button.place(relx=0.40, rely=0.8, relwidth=0.2, relheight=0.05)

        # Adding the done button
        style = ttk.Style()
        style.configure('D.TButton', font=('Helvetica', 17, 'bold'), foreground='#FFB0CA', border='#C2D9DF')
        done_button = ttk.Button(self, text="Done", style='D.TButton',
                                 command=self.display_recommendations)
        done_button.place(relx=0.45, rely=0.85, relwidth=0.1, relheight=0.05)

    def display_recommendations(self) -> None:
        """
        This method will display all the recommendations for the users in a scrolling bar format.
        """

        # Clear the widgets on the screen first
        for widgets in self.winfo_children():
            if widgets not in {self.background_image, self.background_label}:
                widgets.destroy()

        # Displays the title
        title_text = tk.Label(self, text="Recommended Recipes",
                              font=("Helvetica", 30, "bold"), fg='#8f2246', bg="white")
        title_text.place(relx=0.25, rely=0.1, relwidth=0.5, relheight=0.1)

        # Calculates the recommended recipes based on what the user previously inputted
        recipe_info = self.food_graph.recommend_recipes(self.criteria['Chosen Recipe'].lower(),
                                                        self.criteria['Num Recs'],
                                                        self.criteria['Chosen Rec System'].lower(),
                                                        self.criteria['Sentimental'])

        # Displays the recommended recipes in a scrolled text format
        display = st.ScrolledText(self, wrap=tk.WORD, width=30, height=8, font=("Helvetica", 15))

        for recipe in recipe_info:
            display.insert(tk.INSERT, f"""{recipe[1]}: {recipe[0]['name'].upper()}
-----------------------------
Description: {recipe[0]['description']}
Time Needed: {recipe[0]['minutes']} minutes

Ingredients:
{'\n'.join(recipe[0]['ingredients'])}

Steps:
{'\n'.join(recipe[0]['steps'])}


""")
        display.configure(state='disabled')
        display.place(relx=0.1, rely=0.2, relwidth=0.8, relheight=0.7)

    def assign_val(self, options: tk.Listbox, chosen: tk.StringVar, num_rec: tk.Entry, choice: tk.StringVar) -> None:
        """
        Once the user finishes choosing their criteria, it will update the self.criteria variable.
        This method is called when confirm choices button is pressed.

        Preconditions:
            - chosen.get() in ['JACCARD', 'OVERLAP', 'COSINE', 'TANIMOTO', 'SIMRANK', 'SIMRANK_PPLUS', 'PAGERANK']
            - options.get(options.curselection()[0]) in self.food_graph.get_all_recipe_names()
            - num_rec.get().isdigit() and int(num_rec.get()) >= 0
            - choice.get() in {'yes', 'no'}
        """
        if options.curselection() != ():
            self.criteria['Chosen Recipe'] = options.get(options.curselection()[0])
        if chosen.get() != '':
            self.criteria['Chosen Rec System'] = chosen.get()
        if num_rec.get() != 0:
            self.criteria['Num Recs'] = int(num_rec.get())
        if choice.get() != '':
            self.criteria['Sentimental'] = choice.get() == 'yes'


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120,
        'extra-imports': ['tkinter', 'tkinter.scrolledtext', 'graph'],
    })
