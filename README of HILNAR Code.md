- **Author:** Flavio Frascatani
- **Version:** 0.1.2
- **DOI: https://doi.org/10.5281/zenodo.22847128** 
- **Licence:** MIT 
- **Contact:** [flavio.frascatani@gmail.com](mailto:flavio.frascatani@gmail.com)


HILNAR (Hilbertian LLM Narrative Geometry) is designed to process documents inserted in the file corpus.py and analyse (historical) narrative shifts. 
It creates positive pointwise mutual information (PPMI) co-occurrence vectors for each combination of sector and year, projects them into a shared low-dimensional frame, and computes three indices:

- NSI (Narrative Shift Index):         displacement of a certain set of documents between two years
	- NSI_sector (used to calculate NSI_state)
	- NSI_state (final output)

- SAI (Sector Alignment Index):      similarity of a sector's vocabulary to the government sector in                                                       a given year
- TDI (Totalitarian Deterioration Index):   trajectory of SAI over time, per sector and averaged over                                                                non-governmental sectors

The ranges are between [0,1] for the NSI and [-1, 1] for the SAI and TDI

The four sectors chosen to analyse historical narratives are: Government (g), Media (m), Education (e) and Popular culture (p).

### Status and Limitations

As for now HILNAR is a prototype and it must be read as giving suggestive but statistically inconclusive findings as these do not yet pass permutation tests (for the reduced size of the corpus). The following must be mentioned:

- The comparison covers 2019 and 2025 and in the state of Belarus
-  The corpus contains just 24 documents (as automatic retrieval is blocked)
- The documents were selected by hand and generally in pairs (for example the New Year address of 2019 of Lukashenka and the one of 2025 both taken in the same source https://president.gov.by)
- The language of the documents is mainly Russian, representing the fact that only 3% of Belarusians use Belarusian in everyday communications (Posokhin, 2019, p. 73), even if for the language used at home this following percentage rises to 26% (National Statistical Committee of the Republic of Belarus, 2020, p. 44).

- The sign of the change in TDI depends on whether Procrustes rotation is applied per slice. For this corpus and analysis, the rotation is off because all slices share the same context columns, so their vectors are already co-framed, and a per-slice Procrustes rotation removes the very drift the NSI measures. The pooled basis avoids privileging a single slice, although it is weighted  towards Education 2019 (56% of tokens). The default specification is for now `rotate=False`. `specification_grid()` reports both settings, and `fig_robustness.png` shows the full distribution. (These choices were made before inspecting the prototype results of the Beta version).



### Usage of AI

Generative AI tools were used in the preparation of this package:

- The code has been developed by creating a theoretical scheme and after by laying down a simple algorithm.
  It was then expanded and improved (especially for the computational efficiency) by Claude Opus 4.6 (Anthropic). The author reviewed the resulting code.
- The descriptive docstrings (`"""..."""`) were written by Claude Opus 4.7 (Anthropic) to save time. Inline comments (`#`) were written by the author.
- For the special case of Belarus, automatic retrieval was blocked, so the author had to manually copy and paste the websites into the corpus. Claude Opus 4.6 removed navigation elements, image captions, links and related-article blocks as, if inserted, they would create more noise (especially the related article blocks). The consistency of the written paragraphs was then double-checked to avoid differences in text by the model.


### Installation

It requires Python 3.9 or later.

`pip install torch matplotlib`

No other packages are needed. Subword segmentation, deduplication, clustering and resampling are implemented with the standard library and torch. 


### Usage 

If the code doesn't work totally or partially and you are working on Windows, add `-X utf8` after `python` in every command below, so that Cyrillic text is read correctly ( `python -X utf8 run.py`). In case read the top comments of the files (#)

For a quick test to see if the code works, run stage.py with the command written in the comments (#) at the beginning of the file stage.py

If the code is still not running, please write to [flavio.frascatani@gmail.com](mailto:flavio.frascatani@gmail.com) 

### Output

All outputs are written to out/. They are the following:

- results.json
- fig_indices.png
- spec_grid.csv
- fig_robustness.png
- boot.json, perm.json, grid.json

### Files

- hilnar_initial_def_for_run.py (method and definitions)
- run.py                                   (it runs the analysis and writes results and figures)
- stage.py                                (runs one inference stage and saves a checkpoint)
- corpus.py                              (the 24 documents with source URL and description)


### Corpus and copyright

corpus.py contains 24 documents published between 2019 and 2025 by Belarusian government portals (president.gov.by, belta.by, pravo.by), media outlets (sputnik.by, sb.by), educational institutions and cultural websites. Each entry records its source URL and a short description.

The texts remain the property of their respective rights holders. They are included solely to allow the analysis to be reproduced and are **not** covered by the MIT licence of this package. Two sources cannot be checked from a public URL alone: one educational document was supplied as a PDF, and one popular-culture document comes from a Russian website (allfest.ru).


### Citation

If you use this software, please cite:

Frascatani, F. (2026). _HILNAR code for "Between Europe and Russia: Two Centuries of Belarusian Historical Narratives (1795–2025)"_ (Version 0.1.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22847128

### Licence

The code is released under the MIT licence; see the file `LICENSE`. The corpus texts are excluded, as described above.





#### References

Национальный статистический комитет Республики Беларусь. (2020).
Общая численность населения, численность населения по возрасту и полу,
состоянию в браке, уровню образования, национальностям, языку,
источникам средств к существованию по Республике Беларусь:
статистический бюллетень [Total population and population by age and sex,
marital status, education, nationality, language and sources of livelihood
in the Republic of Belarus: Statistical bulletin]. Minsk.
https://www.belstat.gov.by/upload/iblock/471/471b4693ab545e3c40d206338ff4ec9e.pdf

Posokhin, I. (2019). Soft Belarusization: (Re)building of identity or "border reinforcement"? Colloquia Humanistica, 8, 57–78. 

