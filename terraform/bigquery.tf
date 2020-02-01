resource "google_bigquery_dataset" "sparkrecipes" {
  dataset_id                  = "sparkrecipes"
  friendly_name               = "sparkrecipes"
  location                    = "US"
}

