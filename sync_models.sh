# first arg: IP address to grab files from
# second arg: the pem file to use

# this is temporary hack, just to make sure it works and remove
# dependency of folders named "will". change these as necessary
mkdir /home/models  
mkdir /home/logs

ssh-keygen -f "~/.ssh/known_hosts" -R $1
rsync -avz --ignore-existing --remove-source-files --rsh "ssh -i $2 -o \"StrictHostKeyChecking no\"" root@$1:/ubuntu/delphi/models /home/models
#rsync -avz --rsh "ssh -i $2 -o \"StrictHostKeyChecking no\"" root@$1:/ubuntu/delphi/logs /home/logs
