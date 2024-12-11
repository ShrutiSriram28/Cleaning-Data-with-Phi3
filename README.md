# Cleaning Data with Phi3
 
## Overview

This project aims to clean a bike share dataset using Microsoft's Phi3 model. 

## Dataset

### Description

The dataset has 183,888 rows. `clean_testfile_full.csv` has the complete dataset which for efficiency, has been truncated to 2,000 rows as in `clean_testfile.csv`. The dataset has the following attributes:

| Attribute | Description |
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

### Errors Introduced

Simple errors have been introduced to the clean dataset using `corrupt_data.py`. Different types of errors have been introduced to different columns.

* **ride_id:**
    - lower case characters to an all upper case string
    - spaces
* **rideable_type:**
    - typos
    - missing values
* **start_at and end_at:**
    - formatting error (_example: YYYYMMDD hhmmss, YYYY/MM/DD hh-mm-ss, etc._)
    - missing values
* **start_station_name and end_station_name:**
    - spaces
    - missing values
* **start_station_id and end_station_id:**
    - incremented the value by 0.1 or 1
* **start_lat, start_lng, end_lat and end_lng:**
    - modified the value 
    - missing values
* **member_casual:**
    - typos (_example: membr, causual, etc._)

`testfile.csv` contains the truncated dataset infused with 15% error.

## Prompts:

Four different prompts were used to clean the dataset:

* Prompt with just column names (`phi3_no_metadata.py`):
```
prompt = f"""Clean this bike data row:
    ride_id: {row.get('ride_id', '')}
    rideable_type: {row.get('rideable_type', '')}
    started_at: {row.get('started_at', '')}
    ended_at: {row.get('ended_at', '')}
    member_casual: {row.get('member_casual', '')}
    start_station: {row.get('start_station_name', '')}, id={row.get('start_station_id', '')}, lat={row.get('start_lat', '')}, lng={row.get('start_lng', '')}
    end_station: {row.get('end_station_name', '')}, id={row.get('end_station_id', '')}, lat={row.get('end_lat', '')}, lng={row.get('end_lng', '')}
    
    Return only a JSON object with the required format below. The JSON must contain ALL fields:
        "ride_id": "value",
        "rideable_type": "value",
        "started_at": "value",
        "ended_at": "value",
        "start_station_name": "value",
        "start_station_id": "value",
        "end_station_name": "value",
        "end_station_id": "value",
        "start_lat": value,
        "start_lng": value,
        "end_lat": value,
        "end_lng": value,
        "member_casual": "value"

    Do not change values to none, null or nan. 
"""
```

* Prompt with column names and some metadata (`phi3_columns.py`):
```
prompt = f"""Clean this bike data row:
    ride_id: {row.get('ride_id', '')}
    rideable_type: {row.get('rideable_type', '')}
    started_at: {row.get('started_at', '')}
    ended_at: {row.get('ended_at', '')}
    member_casual: {row.get('member_casual', '')}
    start_station: {row.get('start_station_name', '')}, id={row.get('start_station_id', '')}, lat={row.get('start_lat', '')}, lng={row.get('start_lng', '')}
    end_station: {row.get('end_station_name', '')}, id={row.get('end_station_id', '')}, lat={row.get('end_lat', '')}, lng={row.get('end_lng', '')}
    
    Return only a JSON object with the required format below. The JSON must contain ALL fields:
        "ride_id": "value",
        "rideable_type": "value",
        "started_at": "value",
        "ended_at": "value",
        "start_station_name": "value",
        "start_station_id": "value",
        "end_station_name": "value",
        "end_station_id": "value",
        "start_lat": value,
        "start_lng": value,
        "end_lat": value,
        "end_lng": value,
        "member_casual": "value"
    
    Rules:
    1. ride_id: Remove spaces and use uppercase. Return one string which is all uppercase. Concatenate all the parts of the string by removing the spaces. Do not leave characters out. Must be one string containing 14 characters without spaces.
    2. rideable_type: Must be 'electric_bike' or 'classic_bike'
    3. dates: 2024-01-01 to 2024-02-01, started_at before ended_at
            - date format must be YYYY-MM-DD hh:mm:ss
    4. member_casual: Must be 'member' or 'casual' in lowercase, nothing else. Correct this field as well.
    5. station_ids: Integers only. Remove decimals
    6. DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THEY MUST BE CORRECTED, FILLED IN CASE OF MISSING VALUES AND RETURNED.
    
    REMEMBER: Use the metadata matches when appropriate to correct station information.
    If no corrections are needed, return the exact original values in the JSON format above.
    Never return null. Never leave fields empty. Always return a complete JSON object.
    Do not return the row with NaN as such. Always fill it up.
"""
```

* Prompt with complete metadata (`phi3_metadata.py`):
```
prompt = f"""Clean this bike data row:
    ride_id: {row.get('ride_id', '')}
    rideable_type: {row.get('rideable_type', '')}
    started_at: {row.get('started_at', '')}
    ended_at: {row.get('ended_at', '')}
    member_casual: {row.get('member_casual', '')}
    start_station: {row.get('start_station_name', '')}, id={row.get('start_station_id', '')}, lat={row.get('start_lat', '')}, lng={row.get('start_lng', '')}
    end_station: {row.get('end_station_name', '')}, id={row.get('end_station_id', '')}, lat={row.get('end_lat', '')}, lng={row.get('end_lng', '')}
    
    Metadata matches found:"""

if start_station_match:
    prompt += f"""
    Start station metadata: name="{start_station_match['station_name']}", id={start_station_match['station_id']}, lat={start_station_match['lat']}, lng={start_station_match['lng']}"""

if end_station_match:
    prompt += f"""
    End station metadata: name="{end_station_match['station_name']}", id={end_station_match['station_id']}, lat={end_station_match['lat']}, lng={end_station_match['lng']}"""

prompt += """

    Return only a JSON object with the required format below. The JSON must contain ALL fields:
        "ride_id": "value",
        "rideable_type": "value",
        "started_at": "value",
        "ended_at": "value",
        "start_station_name": "value",
        "start_station_id": "value",
        "end_station_name": "value",
        "end_station_id": "value",
        "start_lat": value,
        "start_lng": value,
        "end_lat": value,
        "end_lng": value,
        "member_casual": "value"
    
    Rules:
    1. ride_id: Remove spaces and use uppercase. Return one string which is all uppercase. Concatenate all the parts of the string by removing the spaces. Do not leave characters out. Must be one string containing 14 characters without spaces.
    2. rideable_type: Must be 'electric_bike' or 'classic_bike'
    3. dates: 2024-01-01 to 2024-02-01, started_at before ended_at
            - date format must be YYYY-MM-DD hh:mm:ss
    4. member_casual: Must be 'member' or 'casual' in lowercase, nothing else. Correct this field as well.
    5. station_ids: Integers only. Remove decimals
    6. DO NOT OMIT COLUMNS FROM THE ROW. ALL THE COLUMNS MUST BE PRESENT. THEY MUST BE CORRECTED, FILLED IN CASE OF MISSING VALUES AND RETURNED.
    
    REMEMBER: Use the metadata matches when appropriate to correct station information.
    If no corrections are needed, return the exact original values in the JSON format above.
    Never return null. Never leave fields empty. Always return a complete JSON object.
    If the original row has NaN, fill it up using the station metadata.
    Do not return the row with NaN as such. Always fill it up.
"""
```

* Few shot prompt (`phi3_few_shot.py`):
```
prompt = f"""Clean this bike data row:
    ride_id: {row.get('ride_id', '')}
    rideable_type: {row.get('rideable_type', '')}
    started_at: {row.get('started_at', '')}
    ended_at: {row.get('ended_at', '')}
    member_casual: {row.get('member_casual', '')}
    start_station: {row.get('start_station_name', '')}, id={row.get('start_station_id', '')}, lat={row.get('start_lat', '')}, lng={row.get('start_lng', '')}
    end_station: {row.get('end_station_name', '')}, id={row.get('end_station_id', '')}, lat={row.get('end_lat', '')}, lng={row.get('end_lng', '')}
    
    Examples of CORRECT row - Note the exact field names that must be used:
    
    "ride_id": "BBC291376E29C9A1",
    "rideable_type": "classic_bike",
    "started_at": "2024-01-19 20:24:21",
    "ended_at": "2024-01-19 20:34:26",
    "start_station_name": "Florida Ave & R St NW",
    "start_station_id": "31503",
    "end_station_name": "11th & M St NW",
    "end_station_id": "31266",
    "start_lat": 38.9126,
    "start_lng": -77.0135,
    "end_lat": 38.9055785,
    "end_lng": -77.027313,
    "member_casual": "member"
    
    Examples of CORRECT row - Note the exact field names that must be used:

    "ride_id": "DE01351AA3EE520A",
    "rideable_type": "electric_bike",
    "started_at": "2024-01-24 06:01:16",
    "ended_at": "2024-01-24 06:14:36",
    "start_station_name": "11th & Park Rd NW",
    "start_station_id": "31651",
    "end_station_name": "18th & L St NW",
    "end_station_id": "31224",
    "start_lat": 38.931365132,
    "start_lng": -77.028289914,
    "end_lat": 38.903741450919384,
    "end_lng": -77.04245209693909,
    "member_casual": "casual"

    Example of INCORRECT formatting - Do not use these formats:

    "ride_id": "62 4E A0 EB B9 2C 5C D9",
    "rideable_type": "electric bike",     
    "start_at": "2024-01-10 161307",      
    "end_at": "2024-01-10 16:17:08",      
    "start_station_name": "Virginia  Square  Metro  /  Monroe  St  &  9th  St  N",
    "start_station_id": "31024.0",       
    "end_station_name": "Washington-Blvd & 10th St N",  
    "end_station_id": "31026.0",          
    "start_lat": "38.882723927",          
    "start_lng": "-77.103165865",         
    "end_lat": null,                      
    "end_lng": null,                      
    "member_casual": "membr" 

    Example of INCORRECT formatting - Do not use these formats:
    
    "ride_id": "Fa443eB033BaeC9c",
    "rideable_type": "electric bike",
    "start_at": "2024-01-23 183153",
    "end_at": "2024-01-23 184117",
    "start_station_name": "15th  &  P  St  NW",
    "start_station_id": "31201",
    "end_station_name": "14Th & Belmont St Nw",
    "end_station_id": "31119",
    "start_lat": 38.909881353,
    "start_lng": -77.034395814,
    "end_lat": 38.921074,
    "end_lng": -77.031887,
    "member_casual": "causual"             

    Return a JSON object with the cleaned data using EXACTLY the field names shown above:
"""
```

## Note:
File names need to be modified before running the code.