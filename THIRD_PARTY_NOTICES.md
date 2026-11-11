# Third-party notices and rights boundaries

The repository owner has not selected a project-wide software licence. This handoff adds none. Do not infer an MIT, Apache, GPL or Creative Commons licence for the original project code or notes from their availability in a private repository. Third-party data, publications and software retain their own rights and terms.

## Singapore official data

Contains information from the Singapore Department of Statistics, Economic Development Board, National Environment Agency and Singapore Tourism Board sources identified in [DATA_SOURCES.md](DATA_SOURCES.md) and the per-file manifests. Access dates, source update dates, source URLs and hashes are preserved there. The analysis, cleaning, transformations and conclusions are those of this project; they are not attributed to, approved by or endorsed by those agencies.

Datasets obtained through data.gov.sg are governed by the [Singapore Open Data Licence](https://developers.data.gov.sg/open-data-licence). Dataset use through SingStat is also subject to that licence, alongside the applicable [SingStat terms of use](https://www.singstat.gov.sg/terms-of-use). Retain source acknowledgement and the licence link with redistributed data and derived products. The licence has exclusions and conditions; it does not transfer ownership of the agency data or grant all rights in third-party material. Official methodological publications and other non-dataset materials retain their applicable source terms.

The licence and terms links were checked on 11 November 2026. Their current official versions govern subsequent use. This notice does not replace those terms or expand their permissions.

## Hong Kong Observatory calendar material

The Gregorian–Lunar conversion tables are published by the **Hong Kong Observatory (HKO), Government of the Hong Kong Special Administrative Region**. Source: [HKO calendar conversion tables](https://www.hko.gov.hk/en/gts/time/conversion.htm), with yearly table URLs and hashes retained in `data/raw/visitor_arrivals/calendar/manifest.json`. The project extracts dates and derives its own monthly Lunar New Year control; it does not claim ownership of the source tables.

HKO material is governed by its [Intellectual Property Rights Notice and Disclaimer](https://www.hko.gov.hk/en/readme/readme.htm) and [website non-commercial-use conditions](https://www.hko.gov.hk/en/appweb/applink.htm), not by the Singapore Open Data Licence. Those conditions cover attribution, accurate reproduction, notices, disclaimer display and other restrictions. Commercial use has separate requirements. The release retains the official [notice](replication/third_party/notices/hko-intellectual-property-notice.html) and [conditions, including the HKO disclaimer](replication/third_party/notices/hko-noncommercial-conditions.html), with URL/retrieval/hash provenance in their [manifest](replication/third_party/notices/manifest.json). Keep those upstream notices with redistributed HKO material and consult the current source terms before further reuse.

This project contains information extracted from HKO's website. HKO and the Government of the HKSAR have no affiliation with this project and do not endorse its analysis or products. Their information is not a project warranty, and the project does not acquire rights in HKO material. This explanatory attribution is additional to, not a substitute for, the applicable upstream notices and conditions.

## Academic papers and replication material

The cited journal articles, working papers and third-party replication packages remain the property of their authors, publishers or other rights holders under their own terms. A citation, public download URL or local research cache is not treated as a general redistribution licence.

Accordingly, release archives exclude third-party literature PDFs and extracted full text, the AEA replication ZIP, and third-party Stata do-files. Only small provenance/citation manifests and the project's own methodological summaries are included. The literature review does not claim replication of those papers' published empirical results. Users can obtain originals from the primary links in the notes, subject to the sources' access and reuse terms.

## Software dependencies and generated output

Python dependencies and their exact resolved versions are recorded in `uv.lock`; each distribution retains its own licence and notices. They are installed through the package manager rather than copied from the author's virtual environment into this release. Node is used for built-in-module JavaScript validation; no npm dependency tree is bundled. Full visitor-report regeneration uses R, `gt`, `jsonlite` and their dependency licences, with versions recorded in `replication/renv.lock`. Installed R libraries and compiled/native installation artifacts are excluded and must be installed for the reproducing platform. Poppler's `pdftotext` is an external system dependency for full main-analysis source reconciliation; it retains its own licence and is not bundled.

Generated regression tables and figures can incorporate formatting or functionality from installed packages, including Pyfixest, MakeTables, Matplotlib and their dependencies. This notice does not relicense those packages or their assets. Retain any applicable package notices when redistributing software components. Public statistical inputs and scholarly citations do not imply endorsement by their providers, authors or publishers.
