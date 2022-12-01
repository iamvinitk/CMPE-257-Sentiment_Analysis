all: paper.md paper.bib
	pandoc -r markdown+smart+simple_tables+table_captions+yaml_metadata_block \
		--pdf-engine=pdflatex \
		--variable classoption=twocolumn \
		--citeproc \
		--filter scripts/table-filter.py \
		--filter pandoc-tablenos \
        --filter pandoc-eqnos \
		--csl=./styles/ieee.csl \
		-M fignos-warning-level=0 \
		--bibliography=paper.bib \
        -s \
        paper.md -o paper.pdf

tex: paper.md paper.bib
	pandoc -r markdown+smart+simple_tables+table_captions+yaml_metadata_block \
		--pdf-engine=pdflatex \
		--variable classoption=twocolumn \
		--citeproc \
		--filter scripts/table-filter.py \
		--filter pandoc-tablenos \
        --filter pandoc-eqnos \
		--csl=./styles/ieee.csl \
		-M fignos-warning-level=0 \
		--bibliography=paper.bib \
        -s \
        paper.md -o paper.pdf
clean:
	rm *.tex *.pdf

# --template=templates/latex.template \
# --csl=./styles/ieee.csl \
