# Paper: gemma-csm

LaTeX source for the gemma-csm write-up. Self-contained `article` skeleton
(standard packages only) — swap `\documentclass` to a venue template
(NeurIPS/ICML/Interspeech) when targeting submission.

## Build
```bash
latexmk -pdf main.tex      # preferred
# or:
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
No TeX installed? Use Overleaf (upload `main.tex` + `references.bib`) or
`brew install --cask mactex-no-gui` / `brew install tectonic`.

## Status
Skeleton + content scaffolding. `[TODO]`/`[result pending]` mark what's
outstanding. Content source of truth is the vault note
(`gemma-csm-conversational-speech-generation-on-an-encoder-free-multimodal-backbone.md`),
rev. build-log; this LaTeX is the publication format we fill as results land.

## arXiv checklist (when results are in)
- [ ] Results sections filled (RQ2 WER, RQ1 Phase A/B WER + listening)
- [ ] Abstract written last
- [ ] Citations completed in `references.bib`
- [ ] Swap to venue template if desired
- [ ] Categories: primary cs.CL or cs.SD; cross-list eess.AS, cs.LG
- [ ] arXiv account + **endorsement** lined up (main hurdle for independent authors)
- [ ] License chosen (e.g. CC-BY)
