import nfl_data_py as nfl 
scraped_data = nfl.import_combine_data([2000,2024])
print(scraped_data.info())