# NYC population inputs, 2019

These tables cover the 6,493 CBGs selected by `New York city` in the urban sustainability dataset's lookup table. The original initialization samples 10 agents per CBG (64,930 agents).

| File | Contents |
|---|---|
| `cbg_list.csv` | Selected 12-digit CBG identifiers |
| `cbg_extracted_data.csv` | 78 selected ACS demographic count fields plus CBG identifier |
| `cbg_context.csv` | CBG identifier, city, year, and home-CBG context |
| `cbg_geographic_data.csv` | CBG identifier, land/water area, latitude and longitude |

## Sources

1. [SafeGraph Open Census Data, 2019 ACS 5-year estimates](https://docs.safegraph.com/docs/open-census-data). The demographic counts and CBG geographic metadata are selected from this release.
2. Liu, Y., Hui, P., Li, T., Ding, J., Xi, Y., Li, Y., et al. (2023). [A Satellite Imagery Dataset for Long-Term Sustainable Development in US Cities](https://doi.org/10.6084/m9.figshare.23936787.v1). Figshare, version 1. The city lookup and home-CBG context are derived from this release.

The urban records were selected using `City Name = New York city` and year 2019. The city's `CBG Code` values were matched to `census_block_group` in Open Census to select the corresponding demographic records. The selected tables were then joined by CBG identifier. See [the initialization instructions](../../../agent_initialization/README.md) for the source files and pipeline commands.

## Regeneration

Run the archive-based command in [the initialization instructions](../../../agent_initialization/README.md) using the two downloaded source archives. The included tables can also be passed directly to `initialize_agents.py`. Generated JSON files use the field names consumed by TIMA's policy-generation and simulation code.
