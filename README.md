# Web Science and Engineering
Repository for the project of the course Web Science and Engineering DSAIT4055.
You can read or download the full project report here:
[https://emihalache.github.io/wse/](https://emihalache.github.io/wse/)  



## Set up the Project
1) Ensure you have poetry installed, if not run:
```
pipx install poetry
```
2) Create the environment with:
```
poetry install
```
3) To run the code, execute the following command:
```
poetry run python main.py
```
4) If you want to enter the venv, execute the following command:
```
source $(poetry env info --path)/bin/activate
```
Then you can run the script directly with:
```
python main.py 
```
To exit out of the poetry virtual environment run:
```
deactivate
```