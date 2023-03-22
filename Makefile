format:
	isort --sl tsne_modifications/
	black tsne_modifications/
	flake8 --max-line-length=120 tsne_modifications/ --select E,W,C,F401,N