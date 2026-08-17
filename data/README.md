# data/

## corpus/

Raw source documents used to build the CareDesk retrieval corpus. The
contents of `data/corpus/` are gitignored — only the manifest is
committed, so the repository always knows what *should* be present
without storing the documents themselves.

### manifest.json schema

`data/corpus/manifest.json` is a JSON array of entries, one per source
document:

| Field                | Type   | Description                                                                                              |
|-----------------------|--------|------------------------------------------------------------------------------------------------------------|
| `doc_id`             | string | Stable unique identifier for the document.                                                                |
| `filename`           | string | Filename of the document under `data/corpus/`.                                                            |
| `source_type`        | string | One of: `policy_pdf`, `faq_markdown`, `medication_leaflet`, `resolved_ticket`, `staff_runbook`.           |
| `persona_visibility` | string | Who the document may be surfaced to: `patient`, `staff`, or `both`.                                       |
| `provenance`         | string | Where the document came from (source system, URL, or synthetic-generation note).                          |
| `added_date`         | string | ISO 8601 date (`YYYY-MM-DD`) the document was added to the corpus.                                        |

Example entry:

```json
{
  "doc_id": "faq-appointment-reschedule-001",
  "filename": "faq-appointment-reschedule-001.md",
  "source_type": "faq_markdown",
  "persona_visibility": "patient",
  "provenance": "synthetic, authored for CareDesk",
  "added_date": "2026-08-17"
}
```

The manifest starts empty and is populated as documents are added to
the corpus during ingestion work.
