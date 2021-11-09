cd /home/simon/CDE_UBS/thesis/;
git pull;
git status;
find . -size +20M | sed 's|^\./||g' | cat > .gitignore;
".DS_Store" >> .gitignore;
".ipynb_checkpoints" >> .gitignore;
git add .;
git commit -m "automatic commit";
git push -u origin main;
