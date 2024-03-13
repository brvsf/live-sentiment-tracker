from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader(sentence):
    """
    Function to print sentiment and return overall sentiment
    """
    # Instanciating SentimentIntensityAnalyzer object
    vader_model = SentimentIntensityAnalyzer()

    # Creating sentiment dict from vader model
    sentiment_dict = vader_model.polarity_scores(sentence)

    # Printing sentiment percentage
    print(f"This sentence is: {sentiment_dict['pos']*100} % positive")
    print(f"{sentiment_dict['neu']*100} % neutral")
    print(f"{sentiment_dict['neg']*100} % negative")

    # Conditional to return overall sentiment of the sentence
    if sentiment_dict['compound'] >= 0.05:
        overall_sentiment = 'The overall sentiment is Positive 😊'
        return overall_sentiment

    if sentiment_dict['compound'] <= -0.05:
        overall_sentiment = 'The overall sentiment is Negative 😔'
        return overall_sentiment

    else:
        overall_sentiment = 'The overall sentiment is Neutral 😐'
        return overall_sentiment


def vader_scores(sentence):
    """
    Function to return the dict with the sentiments
    """
    vader_model = SentimentIntensityAnalyzer()

    sentiment_dict = vader_model.polarity_scores(sentence)

    return sentiment_dict
