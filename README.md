# major_project
git init
git branch -M main
"`.venv/`nout/`nrun_log.txt" | Out-File -Encoding utf8 .gitignore   # don't upload the venv or outputs
git add run.py requirements.txt .gitignore
git commit -m "Add analysis script and requirements"
git remote add origin https://github.com/Amol6991/major_project.git
