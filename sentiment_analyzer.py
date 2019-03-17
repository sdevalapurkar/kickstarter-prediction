from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
import math

def convertToNumber (s):
  return int.from_bytes(s.encode(), 'little')

print(convertToNumber('foo rab'))

def sentiment_analyzer_scores(sentence):
  score = analyser.polarity_scores(sentence)
  print("{:-<40} {}".format(sentence, str(score)))

sentiment_analyzer_scores("terrible awful hate")
