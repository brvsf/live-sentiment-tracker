from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocess import SlangTranslation

print (f'Creating model functions ⏳')

def vader(sentence):
    """
    Function to print sentiment and return overall sentiment
    """
    # Instanciating SentimentIntensityAnalyzer object
    vader_model = SentimentIntensityAnalyzer()
    sentence = SlangTranslation(sentence).apply_translator(sentence)
    # Creating sentiment dict from vader model
    sentiment_dict = vader_model.polarity_scores(sentence)

    # Printing sentiment percentage
    print(f'Sentence: "{sentence}"')
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
    sentence = SlangTranslation(sentence).apply_translator(sentence)
    sentiment_dict = vader_model.polarity_scores(sentence)
    print(f'Sentence: "{sentence}"')
    return sentiment_dict

sentence_test = "i'm so happy wtf"
print(f'---------------Testing vader scores function---------------')
print(vader_scores(sentence_test))
print(f'------------------Testing vader function------------------')
print(vader(sentence_test))

print (f'Everything is ok ✅')
