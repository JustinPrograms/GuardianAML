# For logging into IBM cloud on the server
echo "$IBMCLOUD_API_KEY" | docker login icr.io -u iamapikey --password-stdin
