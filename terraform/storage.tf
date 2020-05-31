resource "google_storage_bucket" "image-store" {
  name     = "seefood-image-store"
  location = "US-CENTRAL1"
}

resource "google_storage_bucket" "models" {
  name     = "seefood-models"
  location = "US-CENTRAL1"
  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "backup" {
  name     = "seefood-backup"
  location = "US-CENTRAL1"
}

resource "google_storage_bucket_iam_member" "mczech_og_viewer" {
  bucket = google_storage_bucket.models.name
  role = "roles/storage.objectViewer"
  member = "user:mike.czech@ottogroup.com"
}
