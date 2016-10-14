# usage: 
# $ sh node.sh <ip-of-node-to-enter> <path-to-pem>
ssh-keygen -f "/home/$HOME/.ssh/known_hosts" -R $1
ssh ubuntu@$1 -o "StrictHostKeyChecking no" -i $2
