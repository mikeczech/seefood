resource "google_bigquery_dataset" "sparkrecipes" {
  dataset_id                  = "sparkrecipes"
  friendly_name               = "sparkrecipes"
  location                    = "US"
}

resource "google_bigquery_dataset" "flickr" {
  dataset_id                  = "flickr"
  friendly_name               = "flickr"
  location                    = "US"
}
