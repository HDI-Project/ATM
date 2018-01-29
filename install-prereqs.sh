# This will be copied into the tox docker build and run during setup.
apt-get -qq update
apt-get -qq -y install mysql-client libmysqlclient-dev
