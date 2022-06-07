# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
from scrapy import Item, Field


class WebpageTextItem(Item):
    title = Field()
    url = Field()
    depth = Field()
    text = Field()
    added_position = Field()