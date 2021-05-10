#fname=Notes_about_network_propagation
#baname=BachelorThesis
baname=netpropnotes
doc2=stitching_chromatin_puzzle_with_hi-c
#Notes_about_network_propagation.pdf: Notes_about_network_propagation.tex mybib.bib
#	pdflatex $(fname)
#	biber $(fname)
#	pdflatex $(fname)

netpropnotes:
	pdflatex $(baname)
	biber $(baname)
	pdflatex $(baname)

doc2:
	pdflatex $(doc2)
	biber $(doc2)
	pdflatex $(doc2)
