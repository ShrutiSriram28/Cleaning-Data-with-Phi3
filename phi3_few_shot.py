import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import json
from tqdm import tqdm
import gc
from difflib import SequenceMatcher

def create_prompt(row):
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

def clean_csv_with_phi3(csv_path, max_rows=None):
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
        output_file = 'cleaned_data_few_shot.json'
        with open(output_file, 'w') as f:
            json.dump(corrected_rows, f, indent=2)
        
        print(f"\nCleaning complete! Processed {len(corrected_rows)} rows")
        print(f"Results saved to {output_file}")
        
        return corrected_rows
        
    except Exception as e:
        print(f"Error in cleaning process: {str(e)}")
        return None

if __name__ == "__main__":
    clean_csv_with_phi3("testfile_15.csv", max_rows=100)