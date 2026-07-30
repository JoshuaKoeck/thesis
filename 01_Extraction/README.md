# Extraction

this folder holds the code for the extraction from GDELT. To extract the data simply run main.py.
The following constants at the top of the file can be adjusted as needed:

- SCRAPE_TIMEOUT = 5 # in seconds
- MAX_SCRAPE_WORKERS = 30 # how many workers should be used for simultaneous scraping
- DAILY_DIR = "daily" # the tool extracts each day individually and stores a parquet file; this is the path to that folder
- OUTPUT_PATH = "output.parquet" # the path to the final combined parquet file

The tool outputs the parquet files for the individual days into a folder called /daily. Once the tool is done, you can simply merge the parquet files (for instance using pandas) to one big parquet file called output.parquet for the next step.
