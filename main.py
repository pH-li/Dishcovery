"""
This is the main method for the Dishcovery program.
Run this file to start using Dishcovery.

Authors: Yiping Chen, Alan Su, Lily Phan, Defne Eris
Professor: Sadia Sharmin
Course: CSC111, Introduction to Computer Science
Date: April 2024
"""

# Import statements
from gui import DishcoveryGUI

if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 120,
        'extra-imports': ['gui'],
    })

    # Running the app itself
    app = DishcoveryGUI()
    app.mainloop()
