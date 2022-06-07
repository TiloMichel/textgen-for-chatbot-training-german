# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
from typing import Set
from elasticsearch import Elasticsearch, helpers
from hashlib import sha1
import logging

class ElasticsearchPipeline:
    """Elasticsearch Pipeline to persist website text.

    src of some code: https://github.com/jayzeng/scrapy-elasticsearch/blob/master/scrapyelasticsearch/scrapyelasticsearch.py
    Licenced under Apache License, Version 2.0 https://github.com/jayzeng/scrapy-elasticsearch#licence
    """
    settings = None
    es = None
    item_buffer = []
    added_position_counter = 0
    website_text_hashes: Set[str] = set()
    logger = logging.getLogger(__name__)


    @classmethod
    def validate_settings(cls, settings):
        def validate_setting(setting_key):
            if settings[setting_key] is None:
                raise KeyError('%s is not defined in settings.py' % setting_key)

        required_settings = {'ELASTICSEARCH_INDEX'}

        for required_setting in required_settings:
            validate_setting(required_setting)


    @classmethod
    def init_es_client(cls, crawler_settings):
        es_settings = {}
        es_settings['hosts'] = crawler_settings.get('ELASTICSEARCH_SERVERS', 'localhost:9200')
        es_settings['timeout'] = crawler_settings.get('ELASTICSEARCH_TIMEOUT', 60)

        es = Elasticsearch(**es_settings)
        return es


    def close_spider(self, spider):
        if len(self.item_buffer):
            self.send_items()


    @classmethod
    def from_crawler(cls, crawler):
        ext = cls()
        ext.settings = crawler.settings
        cls.validate_settings(ext.settings)
        ext.es = cls.init_es_client(crawler.settings)
        return ext


    def index_item(self, item):
        index_name = self.settings['ELASTICSEARCH_INDEX']
        text_sha1_hash = sha1(item['text'].encode('utf-8')).hexdigest()
        if text_sha1_hash in self.website_text_hashes: # skip duplicates (meaning same website text)
            url = item['url']
            self.logger.warning(f"Duplicate website skipped {url}")
            return

        self.website_text_hashes.add(text_sha1_hash)
        item['added_position'] = self.added_position_counter
        self.added_position_counter += 1
        index_action = {
            '_index': index_name,
            '_source': dict(item)
        }

        self.item_buffer.append(index_action)

        if len(self.item_buffer) >= self.settings.get('ELASTICSEARCH_BUFFER_LENGTH', 500):
            self.send_items()
            self.item_buffer = []


    def send_items(self):
        helpers.bulk(self.es, self.item_buffer) # refresh="wait_for"


    def process_item(self, item, spider):
        self.index_item(item)
        return item
