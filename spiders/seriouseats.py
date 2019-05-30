import scrapy
import re


class SeriousEatsSpider(scrapy.Spider):
    name = "seriouseats"
    base_url = "https://www.seriouseats.com/recipes/topics"

    def start_requests(self):
        yield scrapy.Request(
            url=f"{self.base_url}/ingredient", callback=self.parse_start_page
        )

    def get_pages(self, response):
        pages = []
        for link in response.css("a").xpath("@href").getall():
            match = re.search(r"ingredient\?page=(\d+)", link)
            if match:
                pages.append(int(match.group(1)))
        max_page = max(pages)
        return 1, 1

    def parse_start_page(self, response):
        page_start, page_end = self.get_pages(response)
        for url in [
            f"{self.base_url}/ingredient?page={i}"
            for i in range(page_start, page_end + 1)
        ]:
            yield response.follow(url, self.parse_ingredient_page)

    def parse_ingredient_page(self, response):
        for module in response.css("section#recipes div.module"):
            title = module.css("div.metadata a h4.title").xpath("string(.)").get()
            if title:
                short_description = module.css("div.metadata a p.kicker").xpath("string(.)").get()
                thumbnail = module.css("img").xpath("@data-src").get()
                url = module.css("a").xpath("@href").get()
                request = response.follow(url, self.parse_recipe_page)
                request.meta["data"] = {
                    "title": title,
                    "short_description": short_description,
                    "thumbnail": thumbnail,
                    "url": url,
                }
                yield request
        

    def parse_recipe_page(self, response):
        description = response.css("div.recipe-introduction div.recipe-introduction-body").xpath("string(p[2])").get()
        data = response.meta["data"]
        data["description"] = description
        yield data