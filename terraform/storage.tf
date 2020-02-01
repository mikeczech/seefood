resource "google_storage_bucket" "image-store" {
  name     = "seefood-image-store"
  location = "US-CENTRAL1"
}

resource "google_storage_bucket" "backup" {
  name     = "seefood-backup"
  location = "US-CENTRAL1"
}
