# change to the root directory
cd $(git rev-parse --show-toplevel)

# install sqlite
sudo apt install -y sqlite3

# set up virtual environment; install python dependencies
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt

# set up sqlite database
sqlite3 atm.db < setup/sqlite-setup.sql
