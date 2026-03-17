cd Guardian/inference/endpoints/
  
if [[ "$2" == "-db" ]]; then
      ./host-uvicorn-db.sh
else
      ./host-uvicorn.sh
fi