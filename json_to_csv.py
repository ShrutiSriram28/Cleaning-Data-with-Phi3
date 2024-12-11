import json
import csv
from typing import Dict, List
from collections import OrderedDict


def flatten_json(nested_json: Dict, parent_key: str = '', separator: str = '_') -> OrderedDict:
    """
    Flatten a nested JSON structure into a flat OrderedDict with compound keys.
    
    Args:
        nested_json: The nested JSON dictionary to flatten
        parent_key: The base key to prepend (used in recursion)
        separator: Character to separate nested keys
    
    Returns:
        A flattened OrderedDict with compound keys, maintaining order
    """
    items: List = []
    for key, value in nested_json.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_json(value, new_key, separator).items())
        elif isinstance(value, list):
            if len(value) > 0:
                if isinstance(value[0], dict):
                    raise ValueError("Lists of dictionaries are not supported in flattening")
                items.append((new_key, str(value)))
        else:
            items.append((new_key, value))
    return OrderedDict(items)

def json_to_csv_with_order(json_file_path: str, csv_file_path: str, column_order: List[str]) -> None:
    """
    Convert a JSON file to CSV format with a specified column order.

    Args:
        json_file_path: Path to the input JSON file.
        csv_file_path: Path where the CSV file will be saved.
        column_order: List of column names in the desired order.
    """
    try:
        # Read JSON file with object_pairs_hook to maintain order
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file, object_pairs_hook=OrderedDict)
        
        # Handle both single dict and list of dicts
        if isinstance(data, (dict, OrderedDict)):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON must contain either a dictionary or a list of dictionaries")
        
        # Flatten each dictionary in the list
        flattened_data = []
        for item in data:
            try:
                flattened_item = flatten_json(item)
                flattened_data.append(flattened_item)
            except ValueError as e:
                print(f"Warning: Skipping complex nested structure: {e}")
                continue
        
        if not flattened_data:
            raise ValueError("No valid data to write to CSV")
        
        # Ensure all rows have all columns (fill missing values with empty strings)
        for row in flattened_data:
            for column in column_order:
                if column not in row:
                    row[column] = ""  # Fill missing columns with empty strings
        
        # Write to CSV using the specified column order
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=column_order)
            writer.writeheader()
            writer.writerows(flattened_data)
            
        print(f"Successfully converted {json_file_path} to {csv_file_path} with specified column order")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_file_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

# Specify your input, output files, and column order here
input_file = "cleaned_data.json"
output_file = "cleaned_data.csv"
desired_column_order = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id", "end_station_name", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"
]

# Convert the file with the desired column order
json_to_csv_with_order(input_file, output_file, desired_column_order)
