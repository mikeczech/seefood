import scrapy
import re
import logging


class SparkRecipes(scrapy.Spider):
    name = "sparkrecipes"
    base_url = "https://recipes.sparkpeople.com"
    recipe_start_id = 1600000
    recipe_end_id = 3200000

    custom_settings = {
        "DOWNLOAD_DELAY": 0.2,
        "USER_AGENT": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:38.0) Gecko/20100101 Firefox/38.0",
    }

    def start_requests(self):
        for recipe_id in range(self.recipe_start_id, self.recipe_end_id):
            yield scrapy.Request(
                url=f"{self.base_url}/recipe-detail.asp?recipe={recipe_id}",
                callback=self.parse_recipe_page,
                meta={"recipe_id": recipe_id},
            )

    def extract_servings(self, response):
        elems = response.css("li.servings").xpath("string(.)").getall()
        result = []
        for e in elems:
            m = re.match(r"Servings Per Recipe: ([0-9]+).*", e)
            if m:
                result.append(m.group(1))

        if len(result) > 1:
            logging.warning("Amount of servings is ambiguous")

        if len(result) == 0:
            return None

        return float(result[0])

    def parse_recipe_page(self, response):
        data = {}
        data["id"] = response.meta['recipe_id']
        data["title"] = response.css(".main_box h1::text").get()
        data["image_url"] = response.css("img[itemprop=image]").xpath("@src").get()
        data["ingredients"] = "|||".join(
            response.css("#ingredients ul span").xpath("string(.)").getall()
        )
        data["url"] = f"{self.base_url}/recipe-detail.asp?recipe={response.meta['recipe_id']}"
        data["servings"] = self.extract_servings(response)

        yield response.follow(
            f"{self.base_url}/recipe-calories.asp?recipe={response.meta['recipe_id']}",
            self.parse_calories_page,
            meta={"data": data},
        )

    def parse_calories_page(self, response):
        response.meta["data"]["ingredient_calories"] = "|||".join(
            [
                t.strip()
                for t in response.css("#ingredients::text").getall()
                if "calories of" in t
            ]
        )

        additional_cols = response.xpath(
                f"//table/th/text()|//table/tr/th/text()"
        ).getall()
        for col in additional_cols:
            response.meta["data"][col.strip()] = response.xpath(
                f'//table/tr[th/text()="{col}"]/td/text()|//table/th[text()="{col}"]/following-sibling::td/text()'
            ).get()

        yield response.meta["data"]
