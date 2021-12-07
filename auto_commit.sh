cd /home/simon/CDE_UBS/thesis/;
git pull;
git status;
find . -size +20M | sed 's|^\./||g' | cat > .gitignore;
echo "*.tif" >> .gitignore;
echo "*.jp2" >> .gitignore;
echo ".DS_Store" >> .gitignore;
echo ".ipynb_checkpoints" >> .gitignore;
echo "mac_ignore.sh" >> .gitignore;

echo "auto_commit.sh" >> .gitignore;
git add .;
git commit -m "automatic commit";
git push -u origin main;
