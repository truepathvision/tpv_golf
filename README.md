# TPV_Golf

Cross-platform desktop application for face image matching using InsightFace embeddings and FAISS vector search.

## Features

- Load and inspect source images with pan/zoom controls
- Embed images using InsightFace buffalo_l model
- Build and query FAISS vector databases from image folders
- Top-10 match carousel with similarity scores
- Side-by-side comparison with synced pan/zoom
- Full case management with audit logging
- Admin panel for managing target databases: add/delete individual images or entire folders, view thumbnails of all entries, save/load databases, clear and rebuild

## Setup

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

The InsightFace buffalo_l model (~300 MB) downloads automatically on first run.

## Run

```bash
python src/main.py
```

## Database Management

Use **Admin > Manage Database...** (Ctrl+D) to:

- **Add Image** -- select one or more images to embed and add to the target database
- **Add Folder** -- embed all images in a folder and append to the database
- **Delete Selected** -- remove specific entries from the database
- **Clear Database** -- wipe all entries
- **Save/Load DB** -- persist the FAISS index to disk (`.faiss` + `.json`) or load an existing one

All entries are shown with thumbnails so you can visually verify the database contents.

You can also use **File > Build Vector Database...** to build a new database from scratch from a folder of images.

## Build Distribution

```bash
pyinstaller tpv_golf.spec
```

Output will be in `dist/TPV_Golf/`.
