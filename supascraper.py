# Import necessary libraries
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
import schedule
import time
import logging
# Remove warnings
import warnings
warnings.filterwarnings('ignore')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("etl_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("football_etl")

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Define avg_cols at module level, before the functions
avg_cols = ['tkl_pcnt', 'sot_pcnt', 'sh_per_90', 'sot_per_90', 'g_per_sh', 
            'g_per_sot', 'dist','npxg_per_sh', 'cmp_pcnt', 'succ_pcnt', 'tkld_pcnt']
join_cols = ['squad', 'comp']

def rename_columns_for_sql(df):
    """Renames DataFrame columns for SQL compatibility, ensuring valid column names."""
    special_char_mapping = {'+': '_plus_', '-': '_minus_', '/': '_per_', ' ': '_', '%': '_pcnt', '90s': 'nineties'}
    df.columns = df.columns.astype(str)
    df.columns = [col.lower() for col in df.columns]
    for char, replacement in special_char_mapping.items():
        df.columns = [col.replace(char, replacement) for col in df.columns]
    return df

def convert_age_to_birthdate(age_str):
    if pd.isna(age_str):
        return np.nan
    try:
        years, days = map(int, age_str.split('-'))
        now = datetime.now()
        birth_date = now - timedelta(days=days + 1)
        birth_date = birth_date.replace(year=birth_date.year - years)
        return birth_date.strftime('%y%m%d')  # YYMMDD format
    except Exception:
        return np.nan

def generate_unique_id(player_name, birth_date):
    if pd.isna(player_name) or pd.isna(birth_date):
        return np.nan
    try:
        last_name = player_name.split()[-1][:5].lower()  # First 5 letters of last name
        return f"{last_name}_{birth_date}"
    except Exception:
        return np.nan

def process_table(df, drop_columns, string_cols, float_cols, rename_dict=None):
    df = df.drop(columns=drop_columns, errors='ignore')
    if 'age' in df.columns:
        df['birth_date'] = df['age'].apply(convert_age_to_birthdate)
        df = df.drop(columns=['age'])
    if rename_dict:
        df = df.rename(columns=rename_dict)
    int_cols = [col for col in df.columns if col not in string_cols + float_cols + ['birth_date']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df = df[df['player'] != 'Player']
    df['unique_id'] = df.apply(lambda row: generate_unique_id(row['player'], row['birth_date']), axis=1)
    return df

def process_standard_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'xg', 'npxg', 'xag', 'npxg_plus_xag', 'g_plus_a_minus_pk', 'xg_plus_xag'])

def process_defense_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born','sh'], ['player', 'squad', 'comp'], ['nineties', 'tkl_pcnt'])

def process_passing_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'cmp_pcnt', 'xag', 'xa', 'a_minus_xag'], {'1_per_3': 'pass_into_final_third'})

def process_shooting_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'sot_pcnt', 'sh_per_90', 'sot_per_90', 'g_per_sh', 'g_per_sot', 'dist', 'xg', 'npxg', 'npxg_per_sh', 'g_minus_xg', 'np:g_minus_xg'])

def process_possession_table(df):
    return process_table(df, ['rk', 'nation', 'pos','born'], ['player', 'squad', 'comp'], ['nineties', 'succ_pcnt', 'tkld_pcnt'], {'1_per_3': 'carries_into_final_third'})

def create_performance_table(standard_df, defense_df, shooting_df, passing_df, possession_df):
    """Creates a consolidated performance table from individual stat tables."""
    # List of DataFrames to merge
    dfs = [standard_df, defense_df, shooting_df, passing_df, possession_df]
    
    # Track columns already seen across DataFrames
    seen_cols = set()
    
    # Process each DataFrame
    processed_dfs = []
    for i, (df_name, df) in enumerate(zip(['standard', 'defense', 'shooting', 'passing', 'possession'], dfs)):
        processed_dfs.append(deduplicate_df(df, df_name, seen_cols))
    
    # Merge DataFrames
    performance_df = processed_dfs[0]
    for df_name, df in zip(['defense', 'shooting', 'passing', 'possession'], processed_dfs[1:]):
        performance_df = performance_df.merge(df, on='unique_id', how='inner', suffixes=('', '_dup'))
    
    # Drop duplicated columns
    columns_to_keep = ~performance_df.columns.duplicated()
    performance_df = performance_df.loc[:, columns_to_keep]
    
    return performance_df

def deduplicate_df(df, df_name, seen_cols):
    """Deduplicates DataFrame based on unique_id and removes duplicated features."""
    global avg_cols, join_cols
    
    if df['unique_id'].duplicated().any():
        # Identify numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('unique_id', errors='ignore')
        non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.drop('unique_id', errors='ignore')
        
        # Filter out already seen columns
        new_numeric_cols = [col for col in numeric_cols if col not in seen_cols]
        new_non_numeric_cols = [col for col in non_numeric_cols if col not in seen_cols]
        
        # Create aggregation dictionary
        agg_dict = {}
        for col in new_numeric_cols:
            agg_dict[col] = 'mean' if col in avg_cols else 'sum'
        for col in new_non_numeric_cols:
            agg_dict[col] = (lambda x: ', '.join(x.dropna().unique())) if col in join_cols else 'first'
        
        # Group by unique_id and aggregate
        df_dedup = df[['unique_id'] + list(agg_dict.keys())].groupby('unique_id', as_index=False).agg(agg_dict)
        seen_cols.update(agg_dict.keys())
        return df_dedup
    else:
        new_cols = [col for col in df.columns if col not in seen_cols and col != 'unique_id']
        df_filtered = df[['unique_id'] + new_cols]
        seen_cols.update(new_cols)
        return df_filtered

def extract_fbref_data():
    """Extracts data from FBRef website with proper error handling and retries."""
    urls = {
        'shooting': ('https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats', 'stats_shooting'),
        'passing': ('https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats', 'stats_passing'),
        'defense': ('https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats', 'stats_defense'),
        'standard': ('https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats', 'stats_standard'),
        'possession': ('https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats', 'stats_possession')
    }
    
    dataframes = {}
    max_retries = 3
    
    for key, (url, table_id) in urls.items():
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {key} data (attempt {attempt+1}/{max_retries})...")
                
                # Add headers to mimic browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                html = response.text.replace('<!--', '').replace('-->', '')
                df = pd.read_html(html, attrs={'id': table_id})[0]
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel()
                
                df = df.loc[:, ~df.columns.duplicated()]
                df = rename_columns_for_sql(df)
                dataframes[key] = df
                
                logger.info(f"✅ Successfully fetched and processed {key} data.")
                break
            
            except Exception as e:
                logger.error(f"❌ Error fetching {key} data (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = 5 * (attempt + 1)  # Progressive backoff
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to fetch {key} data after {max_retries} attempts")
    
    return dataframes

def save_to_supabase(data, table_name):
    """Completely replaces all data in the specified Supabase table."""
    try:
        # Method 1: Use a transaction with RPC call to truncate the table
        # This is the most reliable way to completely clear a table
        try:
            logger.info(f"Attempting to truncate {table_name} table...")
            # Using RPC to call a PostgreSQL function that truncates the table
            supabase.rpc(
                'truncate_table', 
                {'table_name': table_name}
            ).execute()
            logger.info(f"Successfully truncated {table_name} table")
        except Exception as e:
            logger.warning(f"RPC truncate failed: {e}. Falling back to alternative method.")
            
            # Method 2: Delete with a condition that matches all rows
            # This is a fallback if the RPC method isn't available
            try:
                logger.info(f"Attempting to delete all rows from {table_name} table...")
                # This will delete all rows in PostgreSQL by matching everything
                supabase.table(table_name).delete().filter('id', 'gte', 0).execute()
                logger.info(f"Successfully deleted all rows from {table_name} table")
            except Exception as e2:
                logger.warning(f"Delete all rows failed: {e2}. Using final fallback method.")
                
                # Method 3: Final fallback - match all text columns with not-null condition
                try:
                    logger.info(f"Using final fallback method to clear {table_name} table...")
                    supabase.table(table_name).delete().filter('unique_id', 'not.is', None).execute()
                    logger.info(f"Successfully cleared {table_name} table using fallback method")
                except Exception as e3:
                    logger.error(f"All methods to clear table failed. Proceeding with insert anyway: {e3}")
        
        # Convert DataFrame to a list of dictionaries
        records = data.to_dict(orient='records')
        
        # Insert records in batches to avoid payload size limits
        batch_size = 500  # Smaller batch size for reliability
        total_batches = (len(records) - 1) // batch_size + 1
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Inserting batch {batch_num}/{total_batches} into {table_name}...")
            
            try:
                supabase.table(table_name).insert(batch).execute()
                logger.info(f"Successfully inserted batch {batch_num}/{total_batches}")
            except Exception as e:
                # If batch insert fails, try one by one
                logger.warning(f"Batch insert failed: {e}. Trying record by record...")
                success_count = 0
                for record in batch:
                    try:
                        supabase.table(table_name).insert(record).execute()
                        success_count += 1
                    except Exception as e2:
                        logger.error(f"Failed to insert record: {e2}")
                
                logger.info(f"Inserted {success_count}/{len(batch)} records individually")
        
        logger.info(f"✅ Successfully saved {len(records)} records to {table_name} table.")
    except Exception as e:
        logger.error(f"❌ Error saving {table_name} data to Supabase: {e}")
        raise

def run_etl_job():
    """Main ETL process function"""
    start_time = datetime.now()
    logger.info(f"🚀 Starting ETL job at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Extract data
        data = extract_fbref_data()
        
        if not data:
            logger.error("No data was extracted. Aborting job.")
            return
        
        # Process individual tables
        processed_tables = {}
        table_processors = {
            'standard': process_standard_table,
            'defense': process_defense_table,
            'passing': process_passing_table,
            'shooting': process_shooting_table,
            'possession': process_possession_table
        }
        
        for key, process_func in table_processors.items():
            if key in data:
                logger.info(f"Processing {key} table...")
                processed_tables[key] = process_func(data[key])
                save_to_supabase(processed_tables[key], key)
        
        # Create and save performance table
        if len(processed_tables) == 5:  # Only if all tables were successfully processed
            logger.info("Creating consolidated performance table...")
            performance_df = create_performance_table(
                processed_tables['standard'],
                processed_tables['defense'],
                processed_tables['shooting'],
                processed_tables['passing'],
                processed_tables['possession']
            )
            save_to_supabase(performance_df, 'performance')
            logger.info(f"Created performance table with {len(performance_df)} records.")
        else:
            logger.warning(f"Only {len(processed_tables)}/5 tables were processed. Skipping performance table creation.")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        logger.info(f"✅ ETL job completed successfully in {duration:.2f} minutes!")
        
    except Exception as e:
        logger.error(f"❌ ETL job failed: {e}", exc_info=True)



# Main execution
if __name__ == "__main__":
    logger.info("Starting Football Data ETL Service")
    try:
        run_etl_job()
    except Exception as e:
        logger.critical(f"Service crashed: {e}", exc_info=True)