import scrapy
import re


class SeriousEatsSpider(scrapy.Spider):
    name = "seriouseats"

    def start_requests(self):
        ingredients = [
            "dairy",
            # "eggs",
            # "fruit",
            # "grains",
            # "meats-and-poultry",
            # "noodles",
            # "nuts",
            # "pasta",
            # "seafood",
            # "tofu",
            # "vegetables",
        ]

        for i in ingredients:
            yield scrapy.Request(
                url=f"https://www.seriouseats.com/recipes/topics/ingredient/{i}",
                callback=self.parse_ingredient_page,
            )

    def parse_ingredient_page(self, response):
        for module in response.css("div.module"):
            title = module.css("div.metadata a h4.title::text").get()
            if title:
                thumbnail = module.css("img").xpath("@data-src").get()
                url = module.css("a").xpath("@href").get()
                yield {
                    'title': title,
                    'thumbnail': thumbnail,
                    'url': url
                }
        # for url in recipe_urls:
        #     yield response.follow(url, self.parse_recipe_page)

    # def parse_recipe_page(self, response):
    #     photo_url = response.css("img.photo").xpath("@src").get()
