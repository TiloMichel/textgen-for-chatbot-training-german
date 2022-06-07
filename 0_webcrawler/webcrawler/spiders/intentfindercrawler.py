from urllib.parse import urlparse

from scrapy.http import HtmlResponse
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy_splash import SplashJsonResponse, SplashRequest, SplashTextResponse
from webcrawler.utils import remove_jsession_token
from webcrawler.items import WebpageTextItem
from justext import justext, get_stoplist


class IntentfindercrawlerSpider(CrawlSpider):
    name = 'intent-finder-bot'
    custom_settings = {
        "CONCURRENT_REQUESTS": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
    }

    rules = (
        Rule(LinkExtractor(), follow=True, callback="parse_item", process_request="use_splash"),
    )


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [kwargs.get('start_url')]
        self.allowed_domains = [urlparse(self.start_url).netloc]
        self.custom_settings["DEPTH_LIMIT"] = kwargs.get('depth', 2)
        self.name = kwargs.get('name', 'intent-finder-bot')


    # Overridden from CrawlSpider
    def parse_start_url(self, response):
        """
        This method is called for each response produced for the URLs in the spiders start_urls attribute.
        It allows to parse the initial responses and must return either an item object, 
        a Request object, or an iterable containing any of them.
        
        src: https://docs.scrapy.org/en/latest/topics/spiders.html?highlight=rule#generic-spiders
        """
        # Workaround: The new request increases the depth, but that shouldn't be the case because these are start_urls (start_urls should be depth = 0, but are depth = 1)
        # Workaround: start_urls are ignored by the rule and ignored by splash
        yield SplashRequest(remove_jsession_token(response.url), self.parse_item, endpoint='render.html', args={'wait': 1})


    def parse_item(self, response):
        paragraphs = justext(
            response.text,
            stoplist=get_stoplist("German"))
        text = ""
        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
                text += f"{paragraph.text}\n\n"
            else:
                self.logger.info(f"The following textblock was classified as boilerplate:\n{paragraph.text}")


        if text: # ignore sites which return no text content from jusText
            yield WebpageTextItem(
                title=response.xpath("//title/text()").extract_first(),
                text=text,
                url=response.url,
                depth=response.meta['depth'])


    # Overridden for scrapy-splash
    # https://github.com/scrapy-plugins/scrapy-splash/issues/92
    def _requests_to_follow(self, response):
        if not isinstance(
                response,
                (HtmlResponse, SplashJsonResponse, SplashTextResponse)):
            return
        seen = set()
        for n, rule in enumerate(self._rules):
            links = [lnk for lnk in rule.link_extractor.extract_links(response)
                     if lnk not in seen]
            if links and rule.process_links:
                links = rule.process_links(links)
            for link in links:
                seen.add(link)
                r = self._build_request(n, link)
                yield rule.process_request(r)


    # scrapy-splash process_request
    # https://github.com/scrapy-plugins/scrapy-splash/issues/92
    def use_splash(self, request):
        new_request = request.replace(url=remove_jsession_token(request.url))
        new_request.meta.update(splash={'args': {'wait': 1,}, 'endpoint': 'render.html'})
        return new_request
