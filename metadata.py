"""
This file contains the Interaction class which stores the reviews for recipes.

Authors: Yiping Chen, Alan Su, Lily Phan, Defne Eris
Professor: Sadia Sharmin
Course: CSC111, Introduction to Computer Science
Date: April 2024
"""
from datetime import date
from typing import Optional, Union
from nltk.sentiment import SentimentIntensityAnalyzer


class Interaction:
    """A class to store the data associated with reviews

    Instance Attributes:
        - rating: The star rating that is given to the corresponding recipe, on a scale from 1-5, inclusive
        - review: An optional string detailing the user's comments on their review
        - interaction_date: The date at which the review was submitted
        - review_sentiment: The computed sentiment score based on the review

    Repesentation Invariatns:
        - 0 <= rating <= 5
        - not review and not interaction_date or review and interaction_date
        - -1.0 <= review_sentiment <= 1.0
    """

    rating: int
    review: Optional[str]
    interaction_date: Optional[date]
    review_sentiment: float

    def __init__(self, rating: int = 0, review: str = None, interaction_date: date = None,
                 load_sentiment: bool = True) -> None:
        self.rating = rating
        self.review = review
        self.interaction_date = interaction_date
        if review and load_sentiment:
            self.review_sentiment = SentimentIntensityAnalyzer().polarity_scores(review)["compound"]
        else:
            self.review_sentiment = 0.0

    def get_weight(self, sentimental: bool = False) -> Union[int, float]:
        """Returns either the 1-5 star rating, if the sentimental argument is False, otherwise returns the
        corresponding review_sentiment instead, if the sentimental argument is True.
        """
        return self.rating if not sentimental else self.review_sentiment


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120,
        'extra-imports': ['datetime', 'nltk.sentiment'],
    })
