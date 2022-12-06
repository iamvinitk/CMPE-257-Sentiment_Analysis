all: paper.md paper.bib
	pandoc \
    --from=markdown \
    --to=latex \
    --template=template.latex \
    --filter=pandoc-crossref \
    --citeproc \
    --bibliography=paper.bib \
    --csl=./styles/ieee.csl \
    --output=paper.pdf \
    paper.md

clean:
	rm *.tex *.pdf