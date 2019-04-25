#/bin/env bash
set -Eeuxo pipefail
declare -r URL="https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip"
declare -r DESTINATION="data"

echo "Saving data to: $DESTINATION"
curl $URL --create-dirs -o "$DESTINATION/traffic-signs-data.zip"
cd $DESTINATION
unzip traffic-signs-data.zip