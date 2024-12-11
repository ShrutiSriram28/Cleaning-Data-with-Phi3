import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import json
from tqdm import tqdm
import gc
from difflib import SequenceMatcher

station_metadata = []

def create_station_metadata(csv_data):
    # Read the CSV data into a DataFrame
    df = pd.read_csv(csv_data)
    
    # Create separate DataFrames for start and end stations
    start_stations = df[['start_station_name', 'start_station_id', 'start_lat', 'start_lng']].copy()
    end_stations = df[['end_station_name', 'end_station_id', 'end_lat', 'end_lng']].copy()
    
    # Rename columns to have common names
    start_stations.columns = ['station_name', 'station_id', 'lat', 'lng']
    end_stations.columns = ['station_name', 'station_id', 'lat', 'lng']
    
    # Combine both DataFrames
    all_stations = pd.concat([start_stations, end_stations], ignore_index=True)
    
    # Group by station name and get the most frequent values for each attribute
    for station_name in all_stations['station_name'].unique():
        station_data = all_stations[all_stations['station_name'] == station_name]
        
        # Get most frequent values
        most_common_id = station_data['station_id'].mode().iloc[0]
        most_common_lat = station_data['lat'].mode().iloc[0]
        most_common_lng = station_data['lng'].mode().iloc[0]
        
        station_metadata.append({
            'station_name': station_name,
            'station_id': most_common_id,
            'lat': float(most_common_lat),
            'lng': float(most_common_lng)
        })

def find_matching_station(station_info, is_start_station=True):
    prefix = 'start_' if is_start_station else 'end_'
    
    # Get the station details from the row
    station_name = str(station_info.get(f'{prefix}station_name', '')).strip().lower()
    station_id = str(station_info.get(f'{prefix}station_id', '')).rstrip('.0')  # Remove .0 from float IDs
    
    try:
        lat = float(station_info.get(f'{prefix}lat', 0))
        lng = float(station_info.get(f'{prefix}lng', 0))
    except (ValueError, TypeError):
        lat = lng = None

    # Go through metadata and find exact matches
    for meta_station in station_metadata:
        # Check station name
        if station_name and station_name != 'nan' and station_name == str(meta_station['station_name']).lower():
            return meta_station
            
        # Check station ID
        if station_id and station_id != 'nan' and station_id == str(meta_station['station_id']).rstrip('.0'):
            return meta_station
            
        # Check coordinates
        if lat is not None and lng is not None and not pd.isna(lat) and not pd.isna(lng):
            if (abs(lat - meta_station['lat']) < 0.0001 and 
                abs(lng - meta_station['lng']) < 0.0001):
                return meta_station
    
    return None

def create_prompt(row):
    # Find matching stations from metadata
    start_station_match = find_matching_station(row, is_start_station=True)
    end_station_match = find_matching_station(row, is_start_station=False)
    
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
    return prompt

# Rest of the functions remain the same
def load_phi3_model():
    torch.random.manual_seed(0)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        device_map=None,
        torch_dtype=torch.float16,  # Use half precision
        trust_remote_code=True,
        attn_implementation="eager"
    ).to(device)
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        model_max_length=2048,
        padding_side='left'
    )
    
    return model, tokenizer, device

def process_single_row(row_data, pipe, generation_args):
    row, row_number = row_data
    
    messages = [
        {"role": "system", "content": "You are a data cleaning expert."},
        {"role": "user", "content": create_prompt(row)}
    ]
    
    try:
        output = pipe(messages, **generation_args)
        response_text = output[0]["generated_text"]
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            try:
                corrected_row = json.loads(response_text[json_start:json_end])
                required_fields = set(row.keys())

                changes = {k: v for k, v in corrected_row.items() if str(v) != str(row[k])}
                if changes:
                    print(f"\nRow {row_number} changes:")
                    for field, new_value in changes.items():
                        print(f"  {field}: {row[field]} → {new_value}")
                return corrected_row
                    
            except json.JSONDecodeError:
                print(f"\nCouldn't parse JSON for row {row_number}")
                return row
                
    except Exception as e:
        print(f"\nError processing row {row_number}: {str(e)}")
        return row

def clean_csv_with_phi3(csv_path, clean_csv_path, max_rows=None):
    try:
        model, tokenizer, device = load_phi3_model()
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        generation_args = {
            "max_new_tokens": 2048,
            "return_full_text": False,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9,
            "num_return_sequences": 1,
            "early_stopping": True
        }
        
        # Read data efficiently
        df = pd.read_csv(csv_path)
        if max_rows:
            df = df.head(max_rows)
        
        # Process rows
        corrected_rows = []
        rows = df.to_dict('records')
        
        for i in tqdm(range(len(rows))):
            result = process_single_row((rows[i], i+1), pipe, generation_args)
            corrected_rows.append(result)
            
            # Memory management
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
            
            # Periodically save progress
            if (i + 1) % 50 == 0:
                temp_output = f'cleaned_data_temp_{i+1}.json'
                with open(temp_output, 'w') as f:
                    json.dump(corrected_rows, f, indent=2)
        
        # Save final results
        output_file = 'cleaned_data_full_metadata.json'
        with open(output_file, 'w') as f:
            json.dump(corrected_rows, f, indent=2)
        
        print(f"\nCleaning complete! Processed {len(corrected_rows)} rows")
        print(f"Results saved to {output_file}")
        
        return corrected_rows
        
    except Exception as e:
        print(f"Error in cleaning process: {str(e)}")
        return None

if __name__ == "__main__":
    create_station_metadata('clean_testfile.csv')
    clean_csv_with_phi3("testfile_15.csv", "clean_testfile.csv", max_rows=100)