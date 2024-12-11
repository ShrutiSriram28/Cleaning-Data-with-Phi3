import pandas as pd # type: ignore
import random
import datetime
from typing import List, Tuple
import re

class DataErrorGenerator:
    def __init__(self, error_probability: float = 0.15, max_empty_percentage: float = 0.03):
        self.error_prob = error_probability
        self.max_empty_percentage = max_empty_percentage
        self.station_variants = {}
        self.valid_start = datetime.datetime(2024, 1, 1, 0, 0, 0)
        self.valid_end = datetime.datetime(2024, 2, 1, 23, 59, 59)
        self.base_datetime_format = '%Y-%m-%d %H:%M:%S'
        self.coordinate_variants = {}
        
    def _generate_invalid_datetime(self) -> str:
        """Generate an invalid datetime outside the valid range."""
        if random.random() < 0.5:
            # Generate date before valid start
            invalid_dt = self.valid_start - datetime.timedelta(
                days=random.randint(1, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
        else:
            # Generate date after valid end
            invalid_dt = self.valid_end + datetime.timedelta(
                days=random.randint(1, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
        return invalid_dt.strftime(self.base_datetime_format)

    def _format_datetime(self, dt_str: str) -> str:
        """Apply random date format transformation."""
        try:
            dt = datetime.datetime.strptime(dt_str, self.base_datetime_format)
            formats = [
                ('%Y/%m/%d %H:%M:%S', lambda x: x),  # Standard with forward slashes
                ('%Y%m%d %H:%M:%S', lambda x: x),    # No date separators
                ('%Y-%m-%d %H%M%S', lambda x: x.replace(':', '')),  # No time separators
                ('%Y-%m-%d-%H-%M-%S', lambda x: x.replace(' ', '-').replace(':', '-')),  # All hyphens
            ]
            chosen_format, transformer = random.choice(formats)
            return dt.strftime(chosen_format)
        except (ValueError, TypeError):
            return dt_str
        
    def _should_introduce_error(self) -> bool:
        return random.random() < self.error_prob
    
    def _modify_ride_id(self, value: str) -> str:
        if pd.isna(value):
            return value
            
        if self._should_introduce_error():
            # Add spaces randomly
            value = ' '.join(value[i:i+2] for i in range(0, len(value), 2))
        
        if self._should_introduce_error():
            # Modify capitalization
            return ''.join(c.lower() if random.random() < 0.3 else c for c in value)
            
        return value
    
    def _modify_rideable_type(self, value: str) -> str:
        if pd.isna(value):
            return value
            
        if self._should_introduce_error():
            errors = {
                'classic_bike': ['class_bike', 'classic_bik', 'clasic_bike', 'classic bike'],
                'electric_bike': ['electrc_bike', 'electric bike', 'eclectic_bike', 'elektric_bike']
            }
            for correct, variants in errors.items():
                if correct in value:
                    return random.choice(variants)
        return value
    
    def _modify_datetime(self, start_dt: str, end_dt: str) -> Tuple[str, str]:
        if pd.isna(start_dt) or pd.isna(end_dt):
            return start_dt, end_dt
            
        try:
            if self._should_introduce_error():
                # Format change
                start_dt = self._format_datetime(start_dt)
                end_dt = self._format_datetime(end_dt)
            
            if self._should_introduce_error():
                # Make end_dt before start_dt with larger time differences
                dt = datetime.datetime.strptime(start_dt, self.base_datetime_format)
                time_difference = random.choice([
                    datetime.timedelta(hours=random.randint(1, 24)),  # Hours difference
                    datetime.timedelta(days=random.randint(1, 7)),    # Days difference
                    datetime.timedelta(weeks=random.randint(1, 4))    # Weeks difference
                ])
                end_dt = (dt - time_difference).strftime(self.base_datetime_format)
                
            return start_dt, end_dt
        except (ValueError, TypeError):
            return start_dt, end_dt

    def _get_station_variants(self, name: str, station_id: str, lat: float = None, lng: float = None) -> dict:
        """Initialize or get station variants including coordinates."""
        if pd.isna(name):
            return {'names': [name], 'ids': [station_id], 'coordinates': [(lat, lng)]}
            
        if name not in self.station_variants:
            # Generate 3-4 coordinate variants for this station
            num_variants = random.randint(3, 4)
            coordinate_variants = []
            
            # If we have initial coordinates, use them as base
            if lat is not None and lng is not None and not (pd.isna(lat) or pd.isna(lng)):
                base_lat, base_lng = lat, lng
                # Generate coordinate variants
                for _ in range(num_variants):
                    lat_variation = random.uniform(-1, 1)
                    lng_variation = random.uniform(-1, 1)
                    coordinate_variants.append((base_lat + lat_variation, base_lng + lng_variation))
            else:
                coordinate_variants = [(lat, lng)]
            
            # Handle station_id carefully
            try:
                if pd.isna(station_id):
                    id_variants = [station_id]
                else:
                    float_id = float(station_id) + 0.1
                    int_id = int(station_id) + 1
                    id_variants = [
                        str(station_id),
                        f"{str(float_id)}",
                        f"{str(int_id)}"
                    ]
            except (ValueError, TypeError):
                id_variants = [str(station_id)]
            
            self.station_variants[name] = {
                'names': [
                    name,
                    name.title(),
                    name.upper(),
                    name.replace('&', 'and'),
                    re.sub(r'\s+', '  ', name)
                ],
                'ids': id_variants,
                'coordinates': coordinate_variants
            }
        return self.station_variants[name]
    
    def _modify_station_name(self, name: str, station_id: str) -> Tuple[str, str]:
        if pd.isna(name) or pd.isna(station_id):
            return name, station_id
        
        variants = self._get_station_variants(name, station_id)
        
        modified_name = name
        modified_id = str(station_id)
        
        if self._should_introduce_error():
            modified_name = random.choice(variants['names'])
        
        if self._should_introduce_error():
            modified_id = random.choice(variants['ids'])
            
        return modified_name, modified_id
    
    def _modify_coordinates(self, lat: float, lng: float, station_name: str, station_id: str) -> Tuple[float, float]:
        """Modify coordinates consistently for the same station."""
        if pd.isna(station_name) or pd.isna(lat) or pd.isna(lng):
            return lat, lng
            
        if self._should_introduce_error():
            # Get or create station variants including coordinates
            variants = self._get_station_variants(station_name, station_id, lat, lng)
            # Choose a random coordinate variant
            return random.choice(variants['coordinates'])
            
        return lat, lng
    
    def _modify_member_type(self, value: str) -> str:
        if pd.isna(value):
            return value
            
        if self._should_introduce_error():
            errors = {
                'member': ['Member', 'MEMBER', 'membr', 'Members'],
                'casual': ['Casual', 'CASUAL', 'causual', 'Casuals']
            }
            for correct, variants in errors.items():
                if correct.lower() == value.lower():
                    return random.choice(variants)
        return value
    
    def introduce_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_errors = df.copy()
        # First, introduce empty values for each column
        for column in df.columns:
            if column != 'ride_id' and column != 'start_station_id' and column != 'end_station_id':  # Skip ride_id
                # Calculate number of empty values to introduce (3% max)
                num_rows = len(df)
                num_empty = int(num_rows * self.max_empty_percentage)
                
                # Randomly select indices for empty values
                empty_indices = random.sample(range(num_rows), num_empty)
                
                # Set those indices to None
                df_with_errors.loc[empty_indices, column] = None

        # Ensure station ID columns are string type
        id_columns = ['start_station_id', 'end_station_id']
        
        for col in id_columns:
            # Convert to string but preserve NaN
            df_with_errors[col] = df_with_errors[col].astype(str)
            # df_with_errors.loc[df_with_errors[col] == 'nan', col] = None
        
        for idx in df_with_errors.index:
            # Modify ride_id
            df_with_errors.at[idx, 'ride_id'] = self._modify_ride_id(df_with_errors.at[idx, 'ride_id'])
            
            # Modify rideable_type if present
            if pd.notna(df_with_errors.at[idx, 'rideable_type']):
                df_with_errors.at[idx, 'rideable_type'] = self._modify_rideable_type(df_with_errors.at[idx, 'rideable_type'])
            
            # Modify dates if both are present
            if pd.notna(df_with_errors.at[idx, 'started_at']) and pd.notna(df_with_errors.at[idx, 'ended_at']):
                start_dt, end_dt = self._modify_datetime(
                    df_with_errors.at[idx, 'started_at'],
                    df_with_errors.at[idx, 'ended_at']
                )
                df_with_errors.at[idx, 'started_at'] = start_dt
                df_with_errors.at[idx, 'ended_at'] = end_dt
            
            # Modify station names and IDs if present
            if pd.notna(df_with_errors.at[idx, 'start_station_name']):
                name, station_id = self._modify_station_name(
                    df_with_errors.at[idx, 'start_station_name'],
                    df_with_errors.at[idx, 'start_station_id']
                )
                df_with_errors.at[idx, 'start_station_name'] = name
                df_with_errors.at[idx, 'start_station_id'] = station_id
            
            if pd.notna(df_with_errors.at[idx, 'end_station_name']):
                name, station_id = self._modify_station_name(
                    df_with_errors.at[idx, 'end_station_name'],
                    df_with_errors.at[idx, 'end_station_id']
                )
                df_with_errors.at[idx, 'end_station_name'] = name
                df_with_errors.at[idx, 'end_station_id'] = station_id
            
            # Modify coordinates if present
            if pd.notna(df_with_errors.at[idx, 'start_lat']):
                lat, lng = self._modify_coordinates(
                    df_with_errors.at[idx, 'start_lat'],
                    df_with_errors.at[idx, 'start_lng'],
                    df_with_errors.at[idx, 'start_station_name'],
                    df_with_errors.at[idx, 'start_station_id']
                )
                df_with_errors.at[idx, 'start_lat'] = lat
                df_with_errors.at[idx, 'start_lng'] = lng
            
            if pd.notna(df_with_errors.at[idx, 'end_lat']):
                lat, lng = self._modify_coordinates(
                    df_with_errors.at[idx, 'end_lat'],
                    df_with_errors.at[idx, 'end_lng'],
                    df_with_errors.at[idx, 'end_station_name'],
                    df_with_errors.at[idx, 'end_station_id']
                )
                df_with_errors.at[idx, 'end_lat'] = lat
                df_with_errors.at[idx, 'end_lng'] = lng
            
            # Modify member/casual status if present
            if pd.notna(df_with_errors.at[idx, 'member_casual']):
                df_with_errors.at[idx, 'member_casual'] = self._modify_member_type(df_with_errors.at[idx, 'member_casual'])
        
        return df_with_errors

def main():
    # Read the CSV file
    df = pd.read_csv('clean_testfile.csv')
    
    # Create error generator with 15% probability for other errors
    # and 3% maximum empty values per column
    error_generator = DataErrorGenerator(error_probability=0.65, max_empty_percentage=0.03)
    
    # Generate errors
    df_with_errors = error_generator.introduce_errors(df)
    
    # Save the result
    df_with_errors.to_csv('testfile.csv', index=False)

if __name__ == "__main__":
    main()