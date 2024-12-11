# Cleaning Data with Phi3
 
## Overview

This project aims to clean a bike share dataset using Microsoft's Phi3 model. 

## Dataset

The dataset has 183,888 rows. For efficiency, the file has been truncated to 2,000 rows. The dataset has the following attributes:

| Attribue | Description |
|------------------|-----------------|
| ride_id | Unique ID for each ride | 
| rideable_type    | Has two values - 'classic_bike' or 'electric_bike' |
| start_at | Starting date of the ride in 'YYYY-MM-DD hh:mm:ss' format |
| end_at | Ending date of the ride in 'YYYY-MM-DD hh:mm:ss' format |
| start_station_name | The name of the station where the ride starts |
| start_station_id | The ID of the station where the ride starts |
| end_station_name | The name of the station where the ride ends |
| end_station_id | The ID of the station where the ride ends |
| start_lat | The latitude of the station where the ride starts |
| start_lat | The longitude of the station where the ride starts |
| end_lat | The latitude of the station where the ride ends |
| end_lat | The longitude of the station where the ride ends |
| member_casual | Has two values - 'member' or 'casual' |