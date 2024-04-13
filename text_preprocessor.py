import config
import csv
import emoji
import unicodedata
import re

class TextPreprocessor():

  def __init__(self):
    self.emoji_dict = self.load_emoji_dict(config.emoji_file_name)
    self.arabic_letters = []
    
  def preprocess(self, text):
    text = self.hashtag_segmentation(text)
    text = self.convert_emoji(text)
    text = self.removals_and_cleaning(text)
    return text

  def convert_emoji(self, text):
    emojis = emoji.emoji_list(text)
    for emo in emojis:
        if emo['emoji'] in self.emoji_dict:
            text = text.replace(emo['emoji'], self.emoji_dict[emo['emoji']])
    return text

  def hashtag_segmentation(self, text):
    pattern = r'#(\w+)'
    def replace(match):
        return match.group(1).replace('_', ' ')
    segmented_text = re.sub(pattern, replace, text)
    return segmented_text

  def removals_and_cleaning(self, text):
    text = self.normalize_characters(text)
    text = self.remove_non_arabic(text)
    text = self.remove_diacritics(text)
    text = self.remove_punctuation(text)
    text = self.remove_elongations(text)
    text = self.remove_consecutive_spaces(text)
    return text

  def normalize_characters(self, text):
    normalized_text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا').replace('ي', 'ى').replace('ئ', 'ى').replace('ة', 'ه')
    return normalized_text
  
  def remove_non_arabic(self, text):
    pattern = r'[^\u0600-\u06FF\s]+'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text
  
  def remove_diacritics(self, text):
    cleaned_text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return cleaned_text
  
  def remove_punctuation(self, text):
    pattern = r'[^\w\s]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text
  
  def remove_elongations(self, text):
    cleaned_text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return cleaned_text

  def remove_consecutive_spaces(self, text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text

  def word_segmentation(self, text):
    # use Farasa
    return text
  
  def load_emoji_dict(self, emoji_file_name):
    emoji_dict = {}
    with open(emoji_file_name, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            emoji, description = row
            emoji_dict[emoji] = description
    return emoji_dict